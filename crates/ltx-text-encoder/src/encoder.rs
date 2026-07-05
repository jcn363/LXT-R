use std::borrow::Borrow;

use tch::nn::Path;
use tch::Tensor;

use crate::config::LTXVTextEncoderConfig;
use crate::embeddings_connector::EmbeddingsConnector;
use crate::embeddings_processor::EmbeddingsProcessor;
use crate::feature_extractor::FeatureExtractor;
use crate::gemma3_text::Gemma3TextModel;
use crate::image_processor::ImageProcessor;
use crate::prompt_enhancement::PromptEnhancer;
use crate::siglip::SigLIPVisionTower;
use crate::t5_encoder::T5EncoderModel;
use crate::tokenizer::LTXVGemmaTokenizer;

pub struct GemmaTextEncoder {
    text_model: Gemma3TextModel,
    tokenizer: LTXVGemmaTokenizer,
    feature_extractor: FeatureExtractor,
    image_processor: ImageProcessor,
    embeddings_processor: EmbeddingsProcessor,
    embeddings_connector: EmbeddingsConnector,
    prompt_enhancer: PromptEnhancer,
    max_text_length: i64,
}

impl GemmaTextEncoder {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &LTXVTextEncoderConfig,
        tokenizer: LTXVGemmaTokenizer,
    ) -> Self {
        let vs = vs.borrow();
        let text_model = Gemma3TextModel::new(vs / "text_model", &config.gemma3);
        let vision_tower = SigLIPVisionTower::new(vs / "vision_tower", &config.siglip);
        let feature_extractor = FeatureExtractor::new(vision_tower);
        let image_processor = ImageProcessor::new(config.siglip.image_size);
        let embeddings_processor = EmbeddingsProcessor::new(
            vs / "embeddings_processor",
            config.gemma3.hidden_size,
            config.gemma3.hidden_size,
        );
        let embeddings_connector = EmbeddingsConnector::new();
        let prompt_enhancer = PromptEnhancer::new();

        Self {
            text_model,
            tokenizer,
            feature_extractor,
            image_processor,
            embeddings_processor,
            embeddings_connector,
            prompt_enhancer,
            max_text_length: config.max_text_length,
        }
    }

    /// Encode text to hidden states using Gemma3.
    /// Returns full sequence: [B, seq_len, hidden_size].
    pub fn encode(&self, text: &str) -> Tensor {
        let enhanced = self.prompt_enhancer.enhance(text);
        let ids = self.tokenizer.encode(&enhanced).unwrap_or_default();
        let input_ids = Tensor::f_from_slice::<i64>(&ids)
            .expect("Failed to create tensor from token IDs")
            .unsqueeze(0);
        self.text_model.forward(&input_ids)
    }

    /// Encode text with raw token IDs.
    pub fn encode_ids(&self, input_ids: &Tensor) -> Tensor {
        self.text_model.forward(input_ids)
    }

    /// Encode image pixels through SigLIP vision tower.
    pub fn encode_image(&self, pixel_values: &Tensor) -> Tensor {
        let processed = self.image_processor.preprocess(pixel_values);
        self.feature_extractor.forward(&processed)
    }

    /// Encode both text and image, then connect their embeddings.
    pub fn encode_multimodal(&self, text: &str, pixel_values: &Tensor) -> Tensor {
        let text_hidden = self.encode(text);
        let vision_hidden = self.encode_image(pixel_values);

        let vision_pooled = self.embeddings_processor.mean_pool(&vision_hidden);

        let text_projected = self.embeddings_processor.forward(&text_hidden.unsqueeze(1));
        let vision_projected = self
            .embeddings_processor
            .forward(&vision_pooled.unsqueeze(1));

        self.embeddings_connector
            .concatenate(&text_projected, &vision_projected)
    }

    pub fn tokenizer(&self) -> &LTXVGemmaTokenizer {
        &self.tokenizer
    }

    pub fn text_model(&self) -> &Gemma3TextModel {
        &self.text_model
    }

    pub fn feature_extractor(&self) -> &FeatureExtractor {
        &self.feature_extractor
    }

    pub fn hidden_size(&self) -> i64 {
        self.text_model.hidden_size()
    }

    pub fn max_text_length(&self) -> i64 {
        self.max_text_length
    }
}

/// T5 text encoder for LTX-Video — text-only, no vision.
pub struct T5TextEncoder {
    model: T5EncoderModel,
    tokenizer: LTXVGemmaTokenizer,
    max_text_length: i64,
}

impl T5TextEncoder {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &crate::config::T5ConfigData,
        tokenizer: LTXVGemmaTokenizer,
        max_text_length: i64,
    ) -> Self {
        let vs = vs.borrow();
        let model = T5EncoderModel::new(vs, config);
        Self {
            model,
            tokenizer,
            max_text_length,
        }
    }

    /// Encode text to hidden states using T5.
    /// Returns: [B, seq_len, d_model]
    pub fn encode(&self, text: &str) -> Tensor {
        let ids = self.tokenizer.encode(text).unwrap_or_default();
        let ids: Vec<i64> = ids.into_iter().take(self.max_text_length as usize).collect();
        let input_ids = Tensor::from_slice(&ids).unsqueeze(0);
        self.model.forward(&input_ids)
    }

    pub fn encode_ids(&self, input_ids: &Tensor) -> Tensor {
        self.model.forward(input_ids)
    }

    pub fn hidden_size(&self) -> i64 {
        self.model.hidden_size()
    }

    pub fn max_text_length(&self) -> i64 {
        self.max_text_length
    }

    pub fn tokenizer(&self) -> &LTXVGemmaTokenizer {
        &self.tokenizer
    }
}
