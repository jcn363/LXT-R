pub struct PromptEnhancer {
    system_prompt: String,
    user_template: String,
}

impl PromptEnhancer {
    pub fn new() -> Self {
        Self {
            system_prompt: String::from(
                "You are a helpful assistant for video generation.",
            ),
            user_template: String::from("<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"),
        }
    }

    /// Enhance a text prompt with system context and chat formatting.
    pub fn enhance(&self, prompt: &str) -> String {
        self.user_template.replace("{prompt}", prompt)
    }

    /// Wrap prompt with system instruction.
    pub fn with_system(&self, prompt: &str) -> String {
        format!(
            "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            self.system_prompt, prompt
        )
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}
