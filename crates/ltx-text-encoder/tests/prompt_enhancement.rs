use ltx_text_encoder::prompt_enhancement::PromptEnhancer;

#[test]
fn test_prompt_enhancer_default() {
    let enhancer = PromptEnhancer::default();
    assert!(!enhancer.system_prompt().is_empty());
}

#[test]
fn test_prompt_enhancer_enhance() {
    let enhancer = PromptEnhancer::new();
    let result = enhancer.enhance("a cat sitting on a table");
    assert!(result.contains("a cat sitting on a table"));
    assert!(result.contains("<bos>"));
    assert!(result.contains("<start_of_turn>user"));
    assert!(result.contains("<end_of_turn>"));
    assert!(result.contains("<start_of_turn>model"));
}

#[test]
fn test_prompt_enhancer_with_system() {
    let enhancer = PromptEnhancer::new();
    let result = enhancer.with_system("a sunset over mountains");
    assert!(result.contains("a sunset over mountains"));
    assert!(result.contains("helpful assistant"));
    // Should have two user turns (system + user)
    assert!(result.matches("<start_of_turn>user").count() == 2);
}

#[test]
fn test_prompt_enhancer_system_prompt() {
    let enhancer = PromptEnhancer::new();
    let sp = enhancer.system_prompt();
    assert!(sp.contains("video generation"));
}

#[test]
fn test_prompt_enhancer_different_prompts() {
    let enhancer = PromptEnhancer::new();
    let r1 = enhancer.enhance("prompt A");
    let r2 = enhancer.enhance("prompt B");
    assert!(r1.contains("prompt A"));
    assert!(r2.contains("prompt B"));
    assert!(r1 != r2);
}
