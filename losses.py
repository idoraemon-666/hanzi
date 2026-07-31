import torch

from digit_writing.phase_normalized_loss import (
    PHASE_WEIGHTS,
    detached_position_metrics,
    phase_normalized_l1,
    position_l1_metrics,
)


COMPOUND_MOVEMENT_SUBPHASES = ("pre_straight", "transition", "post_straight")


def compound_subphase_equal_position_l1(
    prediction,
    target,
    epoch_bounds,
    movement_subphase,
):
    """Keep canonical phase weights but average compound movement subphases equally."""

    metrics = position_l1_metrics(prediction, target, epoch_bounds)
    movement_start, movement_end = epoch_bounds["movement"]
    labels = tuple(movement_subphase)
    if len(labels) != movement_end - movement_start:
        raise ValueError("compound movement subphase length differs from movement bounds")
    if set(labels) != set(COMPOUND_MOVEMENT_SUBPHASES):
        raise ValueError("compound movement subphases differ")
    movement_error = torch.sum(
        torch.abs(
            prediction[:, movement_start:movement_end]
            - target[:, movement_start:movement_end]
        ),
        dim=-1,
    )
    subphase_means = {
        subphase: torch.mean(
            movement_error[
                :,
                torch.tensor(
                    [label == subphase for label in labels],
                    dtype=torch.bool,
                    device=movement_error.device,
                ),
            ]
        )
        for subphase in COMPOUND_MOVEMENT_SUBPHASES
    }
    equal_movement_mean = torch.stack(tuple(subphase_means.values())).mean()
    objective = (
        PHASE_WEIGHTS["stable"] * metrics["stable_mean_l1"]
        + PHASE_WEIGHTS["delay"] * metrics["delay_mean_l1"]
        + PHASE_WEIGHTS["movement"] * equal_movement_mean
        + PHASE_WEIGHTS["hold"] * metrics["hold_mean_l1"]
    )
    return {
        "objective": objective,
        "movement_subphase_equal_mean_l1": equal_movement_mean,
        **{
            f"{subphase}_mean_l1": value
            for subphase, value in subphase_means.items()
        },
    }


def l1_dist(x, y):
    """L1 loss"""
    return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1

def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1

def l1_muscle_act(act, scale):
    l1 = scale * torch.mean(torch.sum(torch.abs(act), dim=-1))
    return l1

def simple_dynamics(act, mrnn, weight=1e-2):

    W_rec, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
    if mrnn.constrained:
        W_rec = mrnn.apply_dales_law(W_rec, W_rec_mask, W_rec_sign)

    if mrnn.activation_name == "softplus":
        derivative = 1 / (1 + torch.exp(-act))
    elif mrnn.activation_name == "relu":
        derivative = torch.where(act > 0, 1., 0.)
    else:
        raise ValueError("Not implemented for activation")

    d_act = torch.mean(derivative, dim=(1, 0))

    update = W_rec * d_act**2
    update = weight * torch.norm(update)

    return update
