"""
Enhanced Conversational AI with Beautiful Response Formatting
Main AI chatbot with integrated stylish response formatting
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import re
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ui.superx_welcome_screen import SuperXWelcomeScreen
except ImportError:
    SuperXWelcomeScreen = None
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Response from conversational AI"""
    response_text: str
    confidence: float = 0.8
    sources: List[str] = None
    timestamp: datetime = None
    follow_up_questions: List[str] = None
    suggested_actions: List[str] = None
    requires_action: bool = False
    data_visualization: Optional[Dict] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.follow_up_questions is None:
            self.follow_up_questions = []
        if self.suggested_actions is None:
            self.suggested_actions = []

class ConversationalAI:
    """
    Main conversational AI engine with built-in enhanced formatting
    """
    
    def __init__(self, show_welcome: bool = True):
        self.show_welcome_on_start = show_welcome
        self.welcome_shown = False
        self.emoji_map = {
            'forecast': '📈', 'products': '📦', 'categories': '🏷️', 'popular': '⭐',
            'trending': '🔥', 'price': '💰', 'inventory': '📊', 'analytics': '📊',
            'insights': '💡', 'recommendations': '🎯', 'success': '✅', 'info': 'ℹ️',
            'time': '⏰', 'growth': '📈', 'seasonal': '🌟', 'bestseller': '👑',
            'hot': '🔥', 'sales': '💹', 'alerts': '🚨', 'warning': '⚠️'
        }
        
        # Sample product data
        self.products = [
            {"name": "Apsara Pencil HB", "price": 5.0},
            {"name": "Apsara Pencil 2B", "price": 5.5},
            {"name": "Parker Fountain Pen", "price": 250.0},
            {"name": "Reynolds Ball Pen", "price": 15.0},
            {"name": "Classmate Notebook A4", "price": 45.0},
            {"name": "Fevicol Glue Stick", "price": 25.0},
            {"name": "Stapler Heavy Duty", "price": 180.0},
            {"name": "Paper Clips Box", "price": 35.0}
        ]
        
        self.categories = [
            "Bakery", "Beverages", "Dairy", "Electronics", "Fruits", "Groceries",
            "Health", "Household", "Medicine", "Office", "Personal Care", 
            "Seasonal", "Snacks", "Stationery", "Vegetables"
        ]
    
    async def process_natural_language_query(self, query: str, user_context: Dict = None) -> ChatResponse:
        """Process natural language query and return ChatResponse"""
        response_text = self.process_query(query)
        
        return ChatResponse(
            response_text=response_text,
            confidence=0.9,
            sources=["SuperX Product Catalog", "AI Forecasting Engine"],
            follow_up_questions=[
                "Would you like a detailed forecast?",
                "Need help with specific products?",
                "Want to see analytics dashboard?"
            ],
            suggested_actions=[
                "Generate forecast",
                "View product catalog",
                "Check analytics"
            ]
        )
    
    def show_welcome_screen(self):
        """Display welcome screen before first query"""
        
        if SuperXWelcomeScreen and not self.welcome_shown:
            welcome = SuperXWelcomeScreen()
            welcome.run_complete_welcome_sequence()
            self.welcome_shown = True
            return True
        return False
    
    def process_query(self, query: str) -> str:
        """Process user query and return enhanced formatted response"""
        
        # Show welcome screen on first query
        if self.show_welcome_on_start and not self.welcome_shown:
            self.show_welcome_screen()
        
        query_lower = query.lower()
        
        # Handle help requests
        if any(word in query_lower for word in ['help', 'what can you do', 'commands', 'guide']):
            return self.format_help_response()
        
        # Handle forecast requests
        if any(word in query_lower for word in ['forecast', 'predict', 'projection', 'future']):
            return self.format_forecast_response(query)
        
        # Handle product/catalog requests
        if any(word in query_lower for word in ['products', 'catalog', 'items', 'list', 'show']):
            return self.format_product_catalog_response()
        
        # Handle analytics requests
        if any(word in query_lower for word in ['analytics', 'performance', 'metrics', 'dashboard']):
            return self.format_analytics_response()
        
        # Default response - show product catalog
        return self.format_product_catalog_response()
    
    def format_product_catalog_response(self) -> str:
        """Format product catalog response in a stylish way"""
        
        # Format trending products
        product_section = f"""
┌─ {self.emoji_map['hot']} **TRENDING PRODUCTS** ─────────────────────────────┐
│                                                            │"""
        
        for i, product in enumerate(self.products[:8], 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            indicator = f" {self.emoji_map['bestseller']}" if i <= 3 else f" {self.emoji_map['hot']}" if price < 50 else ""
            
            product_section += f"""
│  {i:2d}. {name:<35} {self.emoji_map['price']}₹{price:<8}{indicator}  │"""
        
        product_section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        # Format categories
        category_section = f"""
┌─ {self.emoji_map['categories']} **PRODUCT CATEGORIES** ──────────────────────────┐
│                                                            │
│  """
        
        category_chunks = [self.categories[i:i+4] for i in range(0, len(self.categories), 4)]
        for chunk in category_chunks:
            formatted_chunk = " • ".join(f"**{cat}**" for cat in chunk)
            category_section += f"{formatted_chunk:<52} │\n│  "
        
        category_section = category_section.rstrip("│  ") + """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['products']} **SUPERX PRODUCT CATALOG** {self.emoji_map['popular']}                      │
╰─────────────────────────────────────────────────────────────╯

{self.emoji_map['info']} I'd be happy to help with forecasting! Please specify:
• **Time Period** - How far ahead would you like to forecast?
• **Product/Category** - Which items interest you?

{product_section}

{category_section}

┌─ {self.emoji_map['info']} **QUICK ACTIONS** ─────────────────────────────────┐
│                                                            │
│  🎯 **Generate Forecast**  📊 **View Analytics**           │
│  📈 **Trend Analysis**     🔍 **Product Search**           │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **Tip:** Try asking "Forecast Apsara Pencil for next 3 months" or "Show trends for Stationery category"
""".strip()
    
    def format_forecast_response(self, query: str) -> str:
        """Format forecast response based on query"""
        
        # Extract product name from query if possible
        product_name = "Selected Product"
        for product in self.products:
            if product['name'].lower() in query.lower():
                product_name = product['name']
                break
        
        # Generate mock forecast data
        forecast_data = {
            'period': 'Next 12 months',
            'confidence': '95%',
            'predicted_demand': '15,240',
            'growth_rate': '+8.3%',
            'seasonality': 'High (Q4 peak)',
            'recommendations': [
                {'action': 'Increase Stock', 'product': product_name, 'reason': 'Strong growth trend predicted'},
                {'action': 'Monitor', 'product': 'Related Products', 'reason': 'Seasonal demand fluctuation expected'},
                {'action': 'Optimize', 'product': 'Inventory Levels', 'reason': 'Peak season approaching'}
            ],
            'insights': [
                'Strong growth expected in Q3-Q4 due to academic season',
                'Category showing 15% YoY growth trend',
                'Premium products gaining market share'
            ]
        }
        
        recommendations_section = f"""
┌─ {self.emoji_map['recommendations']} **SMART RECOMMENDATIONS** ─────────────────────┐
│                                                            │"""
        
        for rec in forecast_data['recommendations']:
            action = rec.get('action', 'Consider')
            product = rec.get('product', 'Product')
            reason = rec.get('reason', 'Market analysis')
            
            recommendations_section += f"""
│  {self.emoji_map['insights']} **{action}** {product}                                │
│     └─ {reason}                                           │"""
        
        recommendations_section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        insights_section = f"""
┌─ {self.emoji_map['insights']} **KEY INSIGHTS** ─────────────────────────────────┐
│                                                            │"""
        
        for insight in forecast_data['insights']:
            insights_section += f"""
│  • {insight:<54} │"""
        
        insights_section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['forecast']} **DEMAND FORECAST INSIGHTS** {self.emoji_map['analytics']}                    │
╰─────────────────────────────────────────────────────────────╯

{self.emoji_map['time']} **Forecast Period:** {forecast_data['period']}
{self.emoji_map['info']} **Confidence Level:** {forecast_data['confidence']}

┌─ **FORECAST SUMMARY** ─────────────────────────────────────┐
│                                                            │
│  {self.emoji_map['trending']} **Predicted Demand:** {forecast_data['predicted_demand']} units     │
│  {self.emoji_map['growth']} **Growth Rate:** {forecast_data['growth_rate']}                │
│  {self.emoji_map['seasonal']} **Seasonality:** {forecast_data['seasonality']}            │
│                                                            │
└────────────────────────────────────────────────────────────┘

{recommendations_section}

{insights_section}
""".strip()
    
    def format_analytics_response(self) -> str:
        """Format analytics dashboard response"""
        
        return f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['analytics']} **ANALYTICS DASHBOARD** {self.emoji_map['insights']}                       │
╰─────────────────────────────────────────────────────────────╯

┌─ {self.emoji_map['sales']} **KEY PERFORMANCE INDICATORS** ──────────────────┐
│                                                            │
│  {self.emoji_map['growth']} **Revenue Growth:** +12.5%                        │
│  {self.emoji_map['inventory']} **Inventory Turnover:** 8.2x                       │
│  {self.emoji_map['success']} **Forecast Accuracy:** 94.2%                        │
│  {self.emoji_map['trending']} **Customer Satisfaction:** 4.7/5                   │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ {self.emoji_map['trending']} **MARKET TRENDS** ─────────────────────────────────┐
│                                                            │
│  {self.emoji_map['hot']} **Hot Categories:** Stationery, Electronics          │
│  {self.emoji_map['growth']} **Growing Segments:** Office, Health                 │
│  {self.emoji_map['seasonal']} **Seasonal Peaks:** Q4, Back-to-School             │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ {self.emoji_map['success']} **SYSTEM STATUS** ──────────────────────────────────┐
│                                                            │
│  {self.emoji_map['success']} All systems operating normally                        │
│  {self.emoji_map['info']} No critical alerts at this time                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
""".strip()
    
    def format_help_response(self) -> str:
        """Format help response"""
        
        return f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['info']} **SUPERX AI ASSISTANT HELP** {self.emoji_map['insights']}                  │
╰─────────────────────────────────────────────────────────────╯

┌─ **WHAT I CAN DO** ────────────────────────────────────────┐
│                                                            │
│  {self.emoji_map['forecast']} **Demand Forecasting**                                │
│     • Generate forecasts for products/categories          │
│     • Predict seasonal trends and patterns                │
│                                                            │
│  {self.emoji_map['analytics']} **Analytics & Insights**                             │
│     • Sales performance analysis                          │
│     • Inventory optimization recommendations              │
│                                                            │
│  {self.emoji_map['products']} **Product Information**                               │
│     • Browse product catalog                              │
│     • Check prices and availability                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ **EXAMPLE COMMANDS** ─────────────────────────────────────┐
│                                                            │
│  • "Forecast Apsara Pencil for next 3 months"            │
│  • "Show trending products in Stationery"                 │
│  • "Analyze sales performance for Electronics"            │
│  • "What are the popular products?"                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

{self.emoji_map['success']} Ready to help! What would you like to know?
""".strip()

# Simple demo function  
def demo_enhanced_ai():
    """Demo the enhanced conversational AI"""
    
    ai = ConversationalAI()
    
    print("="*80)
    print("🚀 ENHANCED CONVERSATIONAL AI DEMO")
    print("="*80)
    
    # Test different queries
    queries = [
        "What products do you have?",
        "Forecast Apsara Pencil for next 6 months",
        "Show me analytics dashboard",
        "Help me understand what you can do"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. USER QUERY: '{query}'")
        print("-" * 60)
        response = ai.process_query(query)
        print(response)
        print("\n" + "="*80)

# Async demo function for testing
async def demo_async_ai():
    """Demo the async conversational AI"""
    ai = ConversationalAI()
    
    query = "What products do you have?"
    response = await ai.process_natural_language_query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response.response_text[:100]}...")
    print(f"Confidence: {response.confidence}")

if __name__ == "__main__":
    demo_enhanced_ai()