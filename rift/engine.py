"""Single-process federated simulator using actual serialized low-bit messages."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import RIFTModel
from .data import CityData, Windows, group, steps_per_day, source_statistics
from .privacy import SourceTrainer
from .aggregation import TargetConditionedAggregator
from .compression import PredictiveEndpoint, CommunicationLedger, affine_encode, affine_decode, clone
from . import protocol as p


def make_model(city, stats, *, target, device):
    return RIFTModel(seq_len=12, horizon=6, tod_size=city.day_steps,
                     speed_mean=stats[0], speed_std=stats[1], num_nodes=city.nodes,
                     use_private_graph_residual=target, graph_adj=city.graph() if target else None,
                     private_graph_hidden_dim=8, private_graph_num_layers=1,
                     private_graph_topk=4, private_graph_gate_kind='horizon').to(device)


def train_epochs(model, loader, optimizer, device, epochs, mae=False):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            error = model(x) - y
            loss = error.abs().mean() if mae else error.square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite target training loss.')
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model, city, stats, split, device, batch_size=32):
    model.eval()
    interval = city.interval(split)
    loader = DataLoader(Windows(city, interval, stats), batch_size=batch_size, shuffle=False)
    absolute, squared = torch.zeros(6, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
    count = 0
    for x, y in loader:
        prediction = model(x.to(device)).cpu()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite prediction.')
        error = (prediction-y).double()
        absolute += error.abs().sum((0, 1))
        squared += error.square().sum((0, 1))
        count += error.shape[0]*error.shape[1]
    mae, rmse = absolute/count, torch.sqrt(squared/count)
    return dict(split=split, interval=list(interval), windows=len(loader.dataset),
                horizons={str(h): dict(MAE=float(mae[h-1]), RMSE=float(rmse[h-1])) for h in (1, 3, 6)},
                average_MAE=float(mae[[0, 2, 5]].mean()), average_RMSE=float(rmse[[0, 2, 5]].mean()))


def save_checkpoint(path, model, target, stats, config, stage, *, parent_checkpoint=None, parent_config=None):
    # Target-local artifact: never contains source data, private RNG, or source optimizer states.
    payload = dict(format='rift_target_v1', target=target, model_state=clone(model.state_dict()),
                   stats=stats, config=config, stage=stage)
    if parent_checkpoint is not None:
        payload.update(parent_checkpoint=parent_checkpoint, parent_config=parent_config)
    torch.save(payload, path)


def train(args):
    config = vars(args).copy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a working GPU; CPU is not substituted.')
    cities = {c: CityData(args.data_root, c, synthetic=args.smoke) for c in group(args.target)}
    target_city = cities[args.target]
    support_end = args.target_days * target_city.day_steps
    if support_end >= target_city.interval(args.split)[0]:
        raise ValueError('Target support must end before the evaluation interval starts.')
    target_stats = target_city.support_statistics(args.target_days)
    # Canonical public shared initialization: 288-step calendar, random model seed.
    anchor = RIFTModel(12, 6, 288, 0., 1., 1, use_private_graph_residual=False).get_shared_state()
    torch.manual_seed(args.seed)
    target = make_model(target_city, target_stats, target=True, device=device)
    target.set_shared_state(anchor)
    loader = DataLoader(Windows(target_city, (0, support_end), target_stats), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed+17031))
    optimizer = torch.optim.Adam(target.parameters(), lr=0.001)
    sources = [c for c in group(args.target) if c != args.target]
    source_trainers = []
    devices = [device.index or 0] if device.type == 'cuda' else []
    # Client construction must not change target dropout randomness.
    with torch.random.fork_rng(devices=devices):
        for source in sources:
            stats = source_statistics(cities[source], args.stats_dir)
            model = make_model(cities[source], stats, target=False, device=device)
            source_trainers.append(SourceTrainer(model, cities[source], stats, seed=args.seed,
                city_index=group(args.target).index(source), expected_batch=args.source_batch,
                clip=args.clip, sigma=args.sigma))
    sender = [PredictiveEndpoint() for _ in sources]
    receiver = [PredictiveEndpoint() for _ in sources]
    aggregator = TargetConditionedAggregator(len(sources), args.target_weight, args.temperature, args.smoothing)
    ledger = CommunicationLedger()
    shared = clone(anchor)
    specs = p.identity_basis_set({k: tuple(v.shape) for k, v in shared.items()})
    references = {}
    # Four initial recipients: three sources and the target.
    for peer in range(4):
        frame = p.serialize_fp32_frame(shared, specs, kind=p.KIND_FULL_FP32, round_id=255, receiver_id=peer)
        references[peer] = p.decode_fp32_frame(frame, specs, kind=p.KIND_FULL_FP32, round_id=255, receiver_id=peer)
        ledger.add('download', 'initialization_fp32', -1, peer, frame)
    difference = None
    for round_id in range(args.rounds):
        before = clone(shared)
        source_updates, innovations = [], []
        if round_id < args.source_rounds:
            bases = specs if round_id == 0 else p.build_bases_from_global_difference(difference)
            for i, trainer in enumerate(source_trainers):
                if round_id > 0:
                    frame = p.serialize_basis_frame(bases, round_id, i)
                    local_bases = p.decode_basis_frame(frame, bases, round_id, i)
                    ledger.add('download', 'basis_fp32', round_id, i, frame)
                else:
                    local_bases = bases
                coefficients = trainer.train_round(shared, local_bases, round_id, args.source_steps)
                frame = sender[i].encode(coefficients, local_bases, round_id, i)
                update, innovation = receiver[i].decode(frame, bases, round_id, i)
                if any(not torch.equal(sender[i].history[k], receiver[i].history[k]) for k in shared):
                    raise RuntimeError('Source sender/server history diverged.')
                ledger.add('upload', 'source_predictive_q2', round_id, i, frame)
                source_updates.append(update)
                innovations.append(innovation)
        target.set_shared_state(shared)
        train_epochs(target, loader, optimizer, device, args.local_epochs)
        current = clone(target.get_shared_state())
        delta = {k: current[k]-shared[k] for k in shared}
        frame, _ = p.serialize_q2_frame(delta, specs, round_id, p.TARGET_ID)
        decoded_flat = p.decode_q2_frame(frame, specs, round_id, p.TARGET_ID)
        decoded_target = {k: decoded_flat[k].reshape(v.shape) for k, v in shared.items()}
        ledger.add('upload', 'target_symmetric_q2', round_id, p.TARGET_ID, frame)
        candidate = aggregator(shared, decoded_target, source_updates, innovations) if source_updates else {
            k: shared[k]+decoded_target[k] for k in shared}
        peers = range(4) if round_id < args.source_rounds else [p.TARGET_ID]
        decoded_common = None
        for peer in peers:
            if any(not torch.equal(references[peer][k], shared[k]) for k in shared):
                raise RuntimeError('Broadcast reference is stale.')
            frame = affine_encode(candidate, shared, specs, round_id, peer)
            result = affine_decode(frame, references[peer], specs, round_id, peer)
            if decoded_common is not None and any(not torch.equal(result[k], decoded_common[k]) for k in shared):
                raise RuntimeError('Broadcast receivers decoded different states.')
            references[peer] = clone(result)
            decoded_common = result
            ledger.add('download', 'model_affine_q4', round_id, peer, frame)
        shared = decoded_common
        target.set_shared_state(shared)  # Server and clients commit the same decoded state.
        difference = {k: shared[k]-before[k] for k in shared}
    # Explicit full-precision handoff into local final adaptation.
    frame = p.serialize_fp32_frame(shared, specs, kind=p.KIND_FULL_FP32, round_id=254, receiver_id=p.SINK_ID)
    handed = p.decode_fp32_frame(frame, specs, kind=p.KIND_FULL_FP32, round_id=254, receiver_id=p.SINK_ID)
    target.set_shared_state(handed)
    ledger.add('download', 'handoff_fp32', args.rounds, p.SINK_ID, frame)
    save_checkpoint(output/'before_adaptation.pt', target, args.target, target_stats, config, 'before_adaptation')
    adapt(target, loader, device, args.adapt_epochs)
    save_checkpoint(output/'target.pt', target, args.target, target_stats, config, 'adapted')
    metrics = evaluate(target, target_city, target_stats, args.split, device, args.batch_size)
    result = dict(metrics=metrics, communication=ledger.summary(), smoke=args.smoke,
                  source_privacy=dict(mode='noise_off' if args.sigma == 0 else 'gaussian',
                      sigma=args.sigma, clip=args.clip, steps=args.source_rounds*args.source_steps,
                      expected_batch=args.source_batch, statistics_rho_per_source=0.1,
                      sampling_rates={c: args.source_batch/cities[c].population for c in sources}))
    if args.sigma == 0:
        result['source_privacy']['training_epsilon'] = None
        result['source_privacy']['accounting_status'] = 'no_finite_training_dp_bound'
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), metrics=metrics, communication_MB=result['communication']['MB']), indent=2))


def adapt(model, loader, device, epochs):
    ordinary, temporal = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        (temporal if name.startswith('core.base.') else ordinary).append(parameter)
    optimizer = torch.optim.Adam([{'params': ordinary, 'lr': .002}, {'params': temporal, 'lr': .001}])
    train_epochs(model, loader, optimizer, device, epochs, mae=True)


def from_checkpoint(args):
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != 'rift_target_v1':
        raise ValueError('Expected a target checkpoint produced by this release.')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    config = payload['config']
    city = CityData(args.data_root, payload['target'], synthetic=config.get('smoke', False))
    model = make_model(city, payload['stats'], target=True, device=device)
    model.load_state_dict(payload['model_state'], strict=True)
    if args.command == 'adapt':
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        effective_config = {**config, **vars(args), 'evaluation_interval': list(city.interval(args.split))}
        (output/'config.json').write_text(json.dumps(effective_config, indent=2)+'\n', encoding='utf-8')
        torch.manual_seed(config['seed'])
        loader = DataLoader(Windows(city, (0, config['target_days']*city.day_steps), payload['stats']),
                            batch_size=config['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(config['seed']+17031))
        adapt(model, loader, device, args.adapt_epochs)
        save_checkpoint(output/'target.pt', model, city.city, payload['stats'], effective_config,
                        'adapted_from_checkpoint', parent_checkpoint=str(Path(args.checkpoint).resolve()),
                        parent_config=config)
    metrics = evaluate(model, city, payload['stats'], args.split, device, config['batch_size'])
    if args.command == 'adapt':
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))
