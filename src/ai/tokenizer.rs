use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: HashMap<String, i64>,
    reverse_vocab: HashMap<i64, String>,
    next_token_id: i64,
    special_tokens: HashMap<String, i64>,
    hf: Option<tokenizers::Tokenizer>,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTokenizer {
    pub fn from_hf_files(tokenizer_json: &std::path::Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let mut t = Self::new();
        t.hf = Some(tok);
        Ok(t)
    }
    pub fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_token_id: 0,
            special_tokens: HashMap::new(),
            hf: None,
        };

        // Add special tokens
        tokenizer.add_special_token("<|startoftext|>", 0);
        tokenizer.add_special_token("<|endoftext|>", 1);
        tokenizer.add_special_token("<|user|>", 2);
        tokenizer.add_special_token("<|assistant|>", 3);
        tokenizer.add_special_token("<|system|>", 4);
        tokenizer.add_special_token("<|pad|>", 5);
        tokenizer.add_special_token("<|unk|>", 6);

        tokenizer.next_token_id = 1000; // Start regular tokens after special tokens

        // Add common English words and characters
        tokenizer.build_basic_vocab();

        tokenizer
    }

    fn add_special_token(&mut self, token: &str, id: i64) {
        self.vocab.insert(token.to_string(), id);
        self.reverse_vocab.insert(id, token.to_string());
        self.special_tokens.insert(token.to_string(), id);
    }

    fn build_basic_vocab(&mut self) {
        // Add individual characters
        for i in 32..127 { // Printable ASCII
            let ch = char::from_u32(i as u32).unwrap().to_string();
            if !self.vocab.contains_key(&ch) {
                self.vocab.insert(ch.clone(), self.next_token_id);
                self.reverse_vocab.insert(self.next_token_id, ch);
                self.next_token_id += 1;
            }
        }

        // Add common words
        let common_words = vec![
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "a", "an", "this", "that", "these", "those", "I", "you", "he", "she", "it", "we", "they",
            "what", "where", "when", "why", "how", "who", "which", "can", "could", "will", "would",
            "shall", "should", "may", "might", "must", "hello", "hi", "thank", "thanks", "please",
            "yes", "no", "ok", "okay", "good", "bad", "great", "fine", "well", "nice", "sorry",
            "help", "question", "answer", "problem", "solution", "work", "works", "working",
            "computer", "program", "code", "data", "file", "system", "user", "time", "day", "way",
        ];

        for word in common_words {
            if !self.vocab.contains_key(word) {
                self.vocab.insert(word.to_string(), self.next_token_id);
                self.reverse_vocab.insert(self.next_token_id, word.to_string());
                self.next_token_id += 1;
            }
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<i64> {
        if let Some(hf) = &self.hf {
            if let Ok(enc) = hf.encode(text, true) {
                return enc.get_ids().iter().map(|&id| id as i64).collect();
            }
        }
        let mut tokens = Vec::new();
        
        // Simple word-based tokenization (fallback)
        let words = text.split_whitespace().collect::<Vec<_>>();
        
        for word in words {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                let token_id = self.next_token_id;
                self.vocab.insert(word.to_string(), token_id);
                self.reverse_vocab.insert(token_id, word.to_string());
                self.next_token_id += 1;
                tokens.push(token_id);
            }
        }
        
        tokens
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        if let Some(hf) = &self.hf {
            let ids_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            if let Ok(text) = hf.decode(&ids_u32, true) {
                return text;
            }
        }
        tokens
            .iter()
            .filter_map(|&token_id| self.reverse_vocab.get(&token_id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn get_special_token(&self, token: &str) -> Option<i64> {
        self.special_tokens.get(token).copied()
    }

    pub fn encode_chat(&mut self, role: &str, content: &str) -> Vec<i64> {
        let mut tokens = Vec::new();
        
        // Add role token
        let role_token = match role {
            "user" => "<|user|>",
            "assistant" => "<|assistant|>",
            "system" => "<|system|>",
            _ => "<|user|>",
        };
        
        if let Some(token_id) = self.get_special_token(role_token) {
            tokens.push(token_id);
        }
        
        // Add content tokens
        tokens.extend(self.encode(content));
        
        tokens
    }

    pub fn prepare_chat_input(&mut self, messages: &[crate::ai::ChatMessage]) -> Vec<i64> {
        // If HF tokenizer is available, build a simple role-based prompt string and encode
        if let Some(hf) = &self.hf {
            let mut prompt = String::new();
            for m in messages {
                match m.role {
                    crate::ai::MessageRole::System => {
                        prompt.push_str("System: ");
                        prompt.push_str(&m.content);
                        prompt.push('\n');
                    }
                    crate::ai::MessageRole::User => {
                        prompt.push_str("User: ");
                        prompt.push_str(&m.content);
                        prompt.push('\n');
                    }
                    crate::ai::MessageRole::Assistant => {
                        prompt.push_str("Assistant: ");
                        prompt.push_str(&m.content);
                        prompt.push('\n');
                    }
                }
            }
            // Prompt the assistant for the next turn
            prompt.push_str("Assistant: ");
            if let Ok(enc) = hf.encode(prompt, true) {
                return enc.get_ids().iter().map(|&id| id as i64).collect();
            }
            // If HF encoding fails, fall back to basic path below
        }

        let mut all_tokens = Vec::new();
        
        // Add start token
        if let Some(start_token) = self.get_special_token("<|startoftext|>") {
            all_tokens.push(start_token);
        }
        
        // Add messages
        for message in messages {
            let role = match message.role {
                crate::ai::MessageRole::User => "user",
                crate::ai::MessageRole::Assistant => "assistant",
                crate::ai::MessageRole::System => "system",
            };
            
            let message_tokens = self.encode_chat(role, &message.content);
            all_tokens.extend(message_tokens);
        }
        
        // Add assistant token to prompt for response
        if let Some(assistant_token) = self.get_special_token("<|assistant|>") {
            all_tokens.push(assistant_token);
        }
        
        all_tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let mut tokenizer = SimpleTokenizer::new();
        
        let text = "hello world";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = SimpleTokenizer::new();
        
        assert!(tokenizer.get_special_token("<|startoftext|>").is_some());
        assert!(tokenizer.get_special_token("<|endoftext|>").is_some());
        assert!(tokenizer.get_special_token("<|user|>").is_some());
        assert!(tokenizer.get_special_token("<|assistant|>").is_some());
    }

    #[test]
    fn test_chat_encoding() {
        let mut tokenizer = SimpleTokenizer::new();
        
        let tokens = tokenizer.encode_chat("user", "Hello there");
        assert!(!tokens.is_empty());
        
        // Should start with user token
        assert_eq!(tokens[0], tokenizer.get_special_token("<|user|>").unwrap());
    }
}