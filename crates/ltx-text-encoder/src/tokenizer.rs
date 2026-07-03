use tokenizers::Tokenizer;

pub struct LTXVGemmaTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
}

pub type TokenizerError = Box<dyn std::error::Error + Send + Sync>;

impl LTXVGemmaTokenizer {
    pub fn from_file(path: &str, max_length: usize) -> Result<Self, TokenizerError> {
        let tokenizer = Tokenizer::from_file(path)?;
        Ok(Self { tokenizer, max_length })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError> {
        let encoding = self.tokenizer.encode(text, true)?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        ids.truncate(self.max_length);
        Ok(ids)
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<i64>>, TokenizerError> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    pub fn pad_token_id(&self) -> i64 {
        self.tokenizer
            .get_padding()
            .map_or(0, |p| p.pad_id as i64)
    }

    pub fn eos_token_id(&self) -> i64 {
        1
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }
}
