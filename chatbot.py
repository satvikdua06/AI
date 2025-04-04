import torch
from train import GPT, Config
import tiktoken
import re
from typing import List

class Chatbot:
    def __init__(self, model_path: str = "best_model.pth"):
        """Initialize chatbot with trained model and proper tokenizer."""
        try:
            # Initialize configuration and device
            self.config = Config()
            self.device = torch.device(self.config.device)
            
            # Load trained model
            self.model = GPT(self.config).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Initialize tokenizer (must match training)
            self.enc = tiktoken.get_encoding("gpt2")
            self.eot_token = self.enc.eot_token  # End-of-text token
            
            # Conversation tracking
            self.history: List[str] = []
            self.max_history_tokens = self.config.block_size - 100  # Reserve space for response
            
            print(f"Model loaded successfully (vocab size: {self.config.vocab_size})")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def encode(self, text: str) -> List[int]:
        """Tokenize text using the same method as during training."""
        return self.enc.encode_ordinary(text)

    def decode(self, tokens: List[int]) -> str:
        """Convert tokens to text using the same method as during training."""
        return self.enc.decode(tokens)

    def respond(self, user_input: str, max_length: int = 60, 
               temperature: float = 0.4, top_k: int = 40) -> str:
        """Generate response to user input."""
        try:
            # Update conversation history
            self.history.append(f"User: {user_input}")
            if len(self.history) > 4:  # Keep last 2 exchanges (4 messages)
                self.history = self.history[-4:]
            
            # Prepare context
            context = "\n".join(self.history + ["Bot:"])
            input_ids = self.encode(context)
            
            # Add EOT token if used in training
            if hasattr(self, 'eot_token'):
                input_ids.append(self.eot_token)
            
            # Ensure context fits within model's limits
            if len(input_ids) > self.max_history_tokens:
                excess = len(input_ids) - self.max_history_tokens
                self.history = self.history[excess//2:]  # Remove oldest messages
                context = "\n".join(self.history + ["Bot:"])
                input_ids = self.encode(context)
                if hasattr(self, 'eot_token'):
                    input_ids.append(self.eot_token)
            
            if not input_ids:
                return "I didn't understand that."
                
            # Convert to tensor and generate
            input_tensor = torch.tensor([input_ids], 
                                     dtype=torch.long,
                                     device=self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
            
            # Process response
            response_ids = output_ids[0].tolist()[len(input_ids):]
            response = self.decode(response_ids)
            
            # Clean response
            response = response.split('\n')[0].split('Bot:')[-1].strip()
            response = re.sub(r'[^\w\s.,!?\']', '', response)
            
            # Update history
            self.history.append(f"Bot: {response}")
            return response if response else "I'm not sure how to respond."
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "Sorry, I encountered an error."

def run_chatbot():
    print("Initializing chatbot...")
    try:
        bot = Chatbot()
        print("\nChatbot ready! Let's chat (type 'quit' to exit)")
        
        # Verify tokenization
        test_phrases = [
            "Hello world!", 
            "How are you today?",
            "What's the meaning of life?"
        ]
        print("\nRunning tokenization tests:")
        for phrase in test_phrases:
            encoded = bot.encode(phrase)
            decoded = bot.decode(encoded)
            print(f"  '{phrase}' -> {encoded} -> '{decoded}'")
        
        # Chat loop
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                    
                # Generate response with conservative settings
                response = bot.respond(
                    user_input,
                    max_length=30,
                    temperature=0.2,
                    top_k=30
                )
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Failed to start chatbot: {str(e)}")

if __name__ == "__main__":
    run_chatbot()
