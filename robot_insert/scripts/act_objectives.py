"""Common physical XYZ objective and millimetre validation, independent of PACE."""
import torch


def common_scale_loss(prediction, target, normalizer, scale_m=.01,action_weights=None):
    with torch.autocast(prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        physical = normalizer.unnormalize_actions(prediction)
        normalized_target = normalizer.actions(target)
        def average(error):
            if action_weights is None:return error.mean()
            weights=action_weights.to(device=error.device,dtype=torch.float32)
            if weights.shape!=error.shape[:-1] or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
                raise ValueError('Action weights must be finite, nonnegative and contain supervision')
            return (error*weights.unsqueeze(-1)).sum()/(weights.sum()*error.shape[-1])
        position = average((physical[..., :3]-target[..., :3]).abs()/scale_m)
        rotation = average((prediction[..., 3:]-normalized_target[..., 3:]).abs())
        loss = (3/7)*position+(4/7)*rotation
        return loss, {'position_loss': position, 'rotation_l1': rotation,
                      'normalized_l1': average((prediction-normalized_target).abs())}


@torch.inference_mode()
def evaluate_pose_chunks(model, loader, config):
    from ftact import move_batch
    from train_ftact import autocast_context
    previous = model.training
    model.eval()
    losses = l1 = count = 0
    position = torch.zeros(20, dtype=torch.float64)
    angle = torch.zeros_like(position)
    errors = []
    pred_error_sum = pred_steps = 0
    try:
        for batch in loader:
            batch = move_batch(batch, config.device)
            with autocast_context(config):
                if hasattr(model, 'validation_action_chunk'):
                    prediction = model.validation_action_chunk(batch)
                elif hasattr(model, 'validation_predictions'):
                    prediction, auxiliary_sum, auxiliary_steps = model.validation_predictions(batch)
                    pred_error_sum += auxiliary_sum
                    pred_steps += auxiliary_steps
                else:
                    prediction = model.normalized_action_chunk(batch)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite ACT validation prediction')
            target = batch['action'].float()
            loss, values = common_scale_loss(prediction, target, model.normalizer, config.position_scale_m)
            losses += loss.item()*len(target)
            l1 += values['normalized_l1'].item()*len(target)
            physical = model.normalizer.unnormalize_actions(prediction)
            error = (physical[..., :3]-target[..., :3]).double().cpu()
            errors.append(error)
            position += error.norm(dim=-1).sum(0)
            q = torch.nn.functional.normalize(physical[..., 3:].double(), dim=-1)
            target_q = torch.nn.functional.normalize(target[..., 3:].double(), dim=-1)
            dot = (q*target_q).sum(-1).abs().clamp(0, 1)
            angle += torch.rad2deg(2*torch.acos(dot)).sum(0).cpu()
            count += len(target)
        if count != config.eval_samples:
            raise ValueError(f'Validation used {count} anchors; exactly {config.eval_samples} required')
        signed = torch.cat(errors)
        absolute = signed.abs()
        position, angle = position/count, angle/count
        result = {'loss': losses/count, 'l1': l1/count,
            'position_l2_m': position.mean().item(), 'position_l2_mm': position.mean().item()*1000,
            'position_l2_cm': position.mean().item()*100,
            'position_xyz_mae_mm': (absolute.mean((0,1))*1000).tolist(),
            'position_xyz_bias_mm': (signed.mean((0,1))*1000).tolist(),
            'position_xyz_p90_mm': (absolute.flatten(0,1).quantile(.9,dim=0)*1000).tolist(),
            'position_xyz_by_horizon_mm': (absolute.mean(0)*1000).tolist(),
            'rotation_deg': angle.mean().item(), 'rotation_by_horizon_deg': angle.tolist(),
            'position_by_horizon_m': position.tolist(), 'first_step_position_m': position[0].item(),
            'last_step_position_m': position[-1].item(), 'samples': count,
            'valid_action_steps': count*20, 'latent': 'zero', 'action_offset': 1,
            'position_scale_m': config.position_scale_m, 'xyz_frame': 'base_link',
            'best_selection_metric': 'loss'}
        if hasattr(model, 'validation_predictions'):
            mse = pred_error_sum/(pred_steps*6) if pred_steps else None
            result.update(pred_mse=mse, pred_valid_steps=pred_steps,
                pred_loss_weight=config.pred_loss_weight,
                total_loss=losses/count+config.pred_loss_weight*mse if mse is not None else None,
                pred_target='normalized consecutive FT increments t+1:t+20',
                selection_note='loss remains imitation-only for ACT/PACE comparability')
        if hasattr(model, 'validation_metadata'):result.update(model.validation_metadata())
        return result
    finally:
        model.train(previous)
