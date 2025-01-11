import os
from openai import OpenAI
from philosophical_comparison import PhilosophicalComparator

class BuddhismChatbotV2:
    def __init__(self):
        self.comparator = PhilosophicalComparator()
        self.conversation_history = []
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://openrouter.ai/api/v1"
        )
        
    def chat(self, user_input):
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response with comparison-focused system prompt
        response = self._generate_llm_response(user_input)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def _generate_llm_response(self, user_input):
        """Generate response using OpenRouter API with comparison-focused prompt"""
        messages = self.conversation_history.copy()
        
        system_prompt = {
            "role": "system",
            "content": """You are a Buddhist philosophy expert specializing in comparing different schools. 
                        When analyzing concepts, always:
                        1. Identify which school (Theravāda or Mahayana) emphasizes which aspect
                        2. Explain the philosophical reasoning behind each school's perspective
                        3. Highlight key differences in their approaches
                        4. Provide historical context for these differences
                        5. Suggest practical implications of each perspective"""
        }
            
        messages.insert(0, system_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=messages,
                extra_headers={"HTTP-Referer": "http://localhost:3000"},
                stream=False
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            return "I'm sorry, I couldn't generate a response. Please try again."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble generating a response. Please try again."

    # ... (rest of the methods same as original)

if __name__ == "__main__":
    bot = BuddhismChatbotV2()
    print("Buddhism Chatbot V2 initialized. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = bot.chat(user_input)
        print(f"Bot: {response}")
