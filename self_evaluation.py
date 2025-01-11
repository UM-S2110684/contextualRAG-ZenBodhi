import os
from openai import OpenAI
from philosophical_comparison import PhilosophicalComparator

class BuddhismChatbotV1:
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
        
        # Generate initial response
        response = self._generate_llm_response(user_input)
        
        # Check if input is a comparison question
        is_comparison = self._is_comparison_question(user_input)

        # Check if response done comparison at 1st prompt
        is_comparison_done = self._response_contains_school_analysis(response)
        
        if is_comparison and is_comparison_done:
            formatted_response = response
            
        elif is_comparison:
            # Directly analyze school perspectives
            analysis = self._analyze_school_perspectives(user_input)
            formatted_response = f"{response}\n\n[School Analysis]\n{analysis}"

        else:
            formatted_response = response
            
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": formatted_response})
        
        return formatted_response
            
    def _is_comparison_question(self, user_input):
        """Determine if the question is comparing concepts"""
        comparison_keywords = [' vs ', ' versus ', ' compare ', ' contrast ', 
                             ' difference ', ' similar ', ' between ', ' both ',
                             ' each ', ' respective ']
        
        # Check for explicit comparison keywords
        if any(word in user_input.lower() for word in comparison_keywords):
            return True
            
        # Check for implicit comparisons using 'and'
        if ' and ' in user_input.lower():
            parts = user_input.lower().split(' and ')
            if len(parts) > 1:
                # Check if we're comparing two distinct concepts
                if len(parts[0].split()) > 0 and len(parts[1].split()) > 0:
                    return True
                    
        return False
        
    def _response_contains_school_analysis(self, response):
        """Check if response already contains proper school analysis"""
        # Check for both school names
        has_both_schools = ('theravāda' in response.lower() and 
                           'mahāyāna' in response.lower())
        
        # Check for comparison indicators
        comparison_indicators = ['compare', 'contrast', 'difference', 'differences', 
                                'similar', 'similarities', 'approach', 'perspective', 'between']
        
        # Check if response contains at least 2 comparison indicators
        has_comparison = sum(1 for word in comparison_indicators 
                           if word in response.lower()) >= 2
        
        # Only skip analysis if both conditions are met
        return has_both_schools and has_comparison
        
    def _analyze_school_perspectives(self, user_input):
        """Directly analyze which school emphasizes which aspects"""
        analysis_prompt = f"""Analyze which Buddhist school (Theravāda or Mahayana) emphasizes which aspects in this comparison: {user_input}
            Structure your response as:
            1. Theravāda Perspective:
            - Key emphasis
            - Philosophical basis
            - Practical implications

            2. Mahayana Perspective:
            - Key emphasis
            - Philosophical basis
            - Practical implications

            3. Comparative Analysis:
            - Key differences
            - Complementary aspects
            - Historical context"""
        
        return self._generate_llm_response(analysis_prompt, is_self_response=True)
            
        # Normal response flow
        analysis = self._analyze_input(user_input)
        response = self._generate_llm_response(user_input, analysis)
        evaluated_response = self._evaluate_school_perspective(response)
        self.conversation_history.append({"role": "assistant", "content": evaluated_response})
        return evaluated_response
    
    def _generate_llm_response(self, user_input, analysis=None, is_self_response=False):
        """Generate response using OpenRouter API"""
        messages = self.conversation_history.copy()
        
        system_prompt = {
            "role": "system",
            "content": "You are a Buddhist philosophy expert."
        }
        
        if is_self_response:
            system_prompt["content"] += """ When analyzing comparisons between Theravāda and Mahayana:
                1. Clearly identify which school emphasizes which aspect
                2. Explain the philosophical reasoning behind each perspective
                3. Provide historical context for the differences
                4. Suggest practical implications of each approach"""
        
        if analysis:
            system_prompt["content"] += f" Here's the analysis of the user's question: {analysis}"
            
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
    bot = BuddhismChatbotV1()
    print("Buddhism Chatbot V1 initialized. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = bot.chat(user_input)
        print(f"Bot: {response}")
