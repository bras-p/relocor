import relocor


sde_dict = {

    'bs': {
        'sde': relocor.sdes.bs_sde,
        'payoff': relocor.sdes.bs_payoff,
        'batch_payoff': relocor.sdes.bs_batch_payoff
    },

    'multi_bs': {
        'sde': relocor.sdes.multi_bs_sde,
        'payoff': relocor.sdes.multi_bs_payoff,
        'batch_payoff': relocor.sdes.multi_bs_batch_payoff
    },

    'heston': {
        'sde': relocor.sdes.heston_sde,
        'payoff': relocor.sdes.heston_payoff,
        'batch_payoff': relocor.sdes.heston_batch_payoff
    },

    'multi_heston': {
        'sde': relocor.sdes.multi_heston_sde,
        'payoff': relocor.sdes.multi_heston_payoff,
        'batch_payoff': relocor.sdes.multi_heston_batch_payoff
    },

    'fishing': {
        'sde': relocor.sdes.fishing_sde,
        'payoff': relocor.sdes.fishing_payoff,
        'batch_payoff': relocor.sdes.fishing_batch_payoff
    },

}




def get_action_param(action_name, sde):
    if not action_name in ['diag', 'ortho', 'ortho2d']:
        raise ValueError('action_name must be in [diag, ortho, ortho2d], received {}'.format(action_name))
    if action_name == 'diag':
        return relocor.actions.ActionDiag(sde.dim_noise), relocor.actions.BatchActionDiag(sde.dim_noise)
    if action_name == 'ortho':
        return relocor.actions.ActionOrtho(sde.dim_noise), relocor.actions.BatchActionOrtho(sde.dim_noise)
    if action_name == 'ortho2d':
        return relocor.actions.ActionOrtho2d(), relocor.actions.BatchActionOrtho2d()
