include { TRAIN } from '../modules/train.nf'

workflow PARAM_SWEEP {
    def name = params.name == null ? "vae_single_cell_data_integration" : params.name

    ch_input = Channel.fromList([
        tuple( [ id: name, single_end:false ],
        file(params.input_dir, checkIfExists: true))
    ])

    recon_param_values = Channel.fromList( params.recon_param_values )
    contrastive_param_values = Channel.fromList( params.contrastive_param_values )
    kl_param_values = Channel.fromList( params.kl_param_values )
    classifier_param_values = Channel.fromList( params.classifier_param_values )

    ch_input
        .combine(recon_param_values)
        .combine(contrastive_param_values)
        .combine(kl_param_values)
        .combine(classifier_param_values)
        .tap { param_sweep }

    TRAIN( param_sweep )
}
