use ltx_attention::RopeType;
use ltx_norm::RMSNorm;
use ltx_types::TransformerArgs;

use crate::block::BasicAVTransformerBlock;
use crate::model::LTXModel;

pub fn from_config(args: &TransformerArgs, vs: &tch::nn::Path) -> LTXModel {
    let rope_type = match args.rope_type.as_str() {
        "split" => RopeType::Split,
        _ => RopeType::Interleaved,
    };

    let blocks: Vec<BasicAVTransformerBlock> = (0..args.num_layers)
        .map(|i| {
            BasicAVTransformerBlock::new(
                &(vs / "blocks" / i as usize),
                args.hidden_dim,
                args.num_heads,
                args.head_dim,
                args.context_dim,
                rope_type,
            )
        })
        .collect();

    let norm_out = RMSNorm::default_eps(args.hidden_dim, vs.device());
    let proj_out = tch::nn::linear(
        vs / "proj_out",
        args.hidden_dim,
        args.hidden_dim,
        Default::default(),
    );

    LTXModel::new(blocks, norm_out, proj_out)
}
