"""
Enhanced Chatbot Demo - Beautiful Response Formatting
Simple demo showing the enhanced conversational AI in action
"""

import sys
sys.path.append('src')

from ai_chatbot.conversational_ai import ConversationalAI

def main():
    """Demo the enhanced chatbot"""
    
    print("🤖 Welcome to SuperX AI Assistant with Enhanced Formatting!")
    print("=" * 70)
    
    # Initialize the AI
    ai = ConversationalAI()
    
    # Sample queries to demonstrate
    sample_queries = [
        "What products do you have?",
        "Forecast Apsara Pencil HB for next 3 months", 
        "Show me analytics",
        "Help"
    ]
    
    print("\n📋 Sample Queries:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "=" * 70)
    print("🚀 Interactive Mode - Type your questions (or 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for using SuperX AI Assistant!")
                break
            
            if not user_input:
                continue
            
            print(f"\n🤖 SuperX AI:")
            print("-" * 50)
            response = ai.process_query(user_input)
            print(response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using SuperX AI Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()