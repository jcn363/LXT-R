use sentencepiece::SentencePieceProcessor;

pub struct LTXVGemmaTokenizer {
    model: SentencePieceProcessor,
    max_length: usize,
}

pub type TokenizerError = Box<dyn std::error::Error + Send + Sync>;

impl LTXVGemmaTokenizer {
    #[must_use = "caller must handle tokenizer error"]
    pub fn from_file(path: &str, max_length: usize) -> Result<Self, TokenizerError> {
        let model = SentencePieceProcessor::open(path)?;
        Ok(Self { model, max_length })
    }

    #[must_use = "caller must handle tokenization error"]
    pub fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError> {
        let encoding = self.model.encode(text)?;
        let mut ids: Vec<i64> = encoding.iter().map(|piece| piece.id as i64).collect();
        ids.truncate(self.max_length);
        Ok(ids)
    }

    #[must_use = "caller must handle batch tokenization error"]
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<i64>>, TokenizerError> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    pub fn pad_token_id(&self) -> i64 {
        0
    }

    pub fn eos_token_id(&self) -> i64 {
        1
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }
}
