"""
Enhanced Response Formatter for Stylish Chatbot Responses
Creates visually appealing and well-structured responses with emojis, formatting, and styling
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class EnhancedResponseFormatter:
    """
    Enhanced formatter for creating stylish and presentable chatbot responses
    """
    
    def __init__(self):
        self.emoji_map = {
            'forecast': '📈',
            'products': '📦',
            'categories': '🏷️',
            'popular': '⭐',
            'trending': '🔥',
            'price': '💰',
            'discount': '🏷️',
            'inventory': '📊',
            'analytics': '📊',
            'insights': '💡',
            'recommendations': '🎯',
            'alerts': '🚨',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️',
            'time': '⏰',
            'location': '📍',
            'user': '👤',
            'sales': '💹',
            'growth': '📈',
            'decline': '📉',
            'stable': '➡️',
            'seasonal': '🌟',
            'promotion': '🎉',
            'new': '🆕',
            'hot': '🔥',
            'bestseller': '👑',
            'limited': '⏳',
            'exclusive': '💎'
        }
    
    def format_product_catalog_response(self, products: List[Dict], categories: List[str]) -> str:
        """Format product catalog response in a stylish way"""
        
        response = f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['products']} **SUPERX PRODUCT CATALOG** {self.emoji_map['popular']}                      │
╰─────────────────────────────────────────────────────────────╯

{self.emoji_map['info']} I'd be happy to help with forecasting! Please specify:
• **Time Period** - How far ahead would you like to forecast?
• **Product/Category** - Which items interest you?

{self._format_popular_products_section(products)}

{self._format_categories_section(categories)}

{self._format_action_buttons()}
"""
        return response.strip()
    
    def _format_popular_products_section(self, products: List[Dict]) -> str:
        """Format popular products section"""
        
        section = f"""
┌─ {self.emoji_map['hot']} **TRENDING PRODUCTS** ─────────────────────────────┐
│                                                            │"""
        
        for i, product in enumerate(products[:8], 1):
            price_emoji = self.emoji_map['price']
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            
            # Add special indicators
            indicator = ""
            if i <= 3:
                indicator = f" {self.emoji_map['bestseller']}"
            elif price < 50:
                indicator = f" {self.emoji_map['hot']}"
            
            section += f"""
│  {i:2d}. {name:<35} {price_emoji}₹{price:<8}{indicator}  │"""
        
        section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
    def _format_categories_section(self, categories: List[str]) -> str:
        """Format categories section"""
        
        section = f"""
┌─ {self.emoji_map['categories']} **PRODUCT CATEGORIES** ──────────────────────────┐
│                                                            │
│  """
        
        # Format categories in rows of 4
        category_chunks = [categories[i:i+4] for i in range(0, len(categories), 4)]
        
        for chunk in category_chunks:
            formatted_chunk = " • ".join(f"**{cat}**" for cat in chunk)
            section += f"{formatted_chunk:<52} │\n│  "
        
        section = section.rstrip("│  ") + """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
    def _format_action_buttons(self) -> str:
        """Format action buttons section"""
        
        return f"""
┌─ {self.emoji_map['info']} **QUICK ACTIONS** ─────────────────────────────────┐
│                                                            │
│  🎯 **Generate Forecast**  📊 **View Analytics**           │
│  📈 **Trend Analysis**     🔍 **Product Search**           │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **Tip:** Try asking "Forecast Apsara Pencil for next 3 months" or "Show trends for Stationery category"
"""
    
    def format_forecast_response(self, forecast_data: Dict[str, Any]) -> str:
        """Format forecast response in a stylish way"""
        
        response = f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['forecast']} **DEMAND FORECAST INSIGHTS** {self.emoji_map['analytics']}                    │
╰─────────────────────────────────────────────────────────────╯

{self.emoji_map['time']} **Forecast Period:** {forecast_data.get('period', 'Next 12 periods')}
{self.emoji_map['info']} **Confidence Level:** {forecast_data.get('confidence', '95%')}

┌─ **FORECAST SUMMARY** ─────────────────────────────────────┐
│                                                            │
│  {self.emoji_map['trending']} **Predicted Demand:** {forecast_data.get('predicted_demand', 'N/A')} units     │
│  {self.emoji_map['growth']} **Growth Rate:** {forecast_data.get('growth_rate', '+5.2%')}                │
│  {self.emoji_map['seasonal']} **Seasonality:** {forecast_data.get('seasonality', 'Moderate')}            │
│                                                            │
└────────────────────────────────────────────────────────────┘

{self._format_product_recommendations(forecast_data.get('recommendations', []))}

{self._format_insights_section(forecast_data.get('insights', []))}
"""
        return response.strip()
    
    def _format_product_recommendations(self, recommendations: List[Dict]) -> str:
        """Format product recommendations"""
        
        if not recommendations:
            return ""
        
        section = f"""
┌─ {self.emoji_map['recommendations']} **SMART RECOMMENDATIONS** ─────────────────────┐
│                                                            │"""
        
        for rec in recommendations[:3]:
            action = rec.get('action', 'Consider')
            product = rec.get('product', 'Product')
            reason = rec.get('reason', 'Market analysis')
            
            section += f"""
│  {self.emoji_map['insights']} **{action}** {product}                                │
│     └─ {reason}                                           │"""
        
        section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
    def _format_insights_section(self, insights: List[str]) -> str:
        """Format insights section"""
        
        if not insights:
            return ""
        
        section = f"""
┌─ {self.emoji_map['insights']} **KEY INSIGHTS** ─────────────────────────────────┐
│                                                            │"""
        
        for insight in insights[:3]:
            section += f"""
│  • {insight:<54} │"""
        
        section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
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
"""
    
    def format_analytics_response(self, analytics_data: Dict[str, Any]) -> str:
        """Format analytics response"""
        
        response = f"""
╭─────────────────────────────────────────────────────────────╮
│  {self.emoji_map['analytics']} **ANALYTICS DASHBOARD** {self.emoji_map['insights']}                       │
╰─────────────────────────────────────────────────────────────╯

{self._format_kpi_section(analytics_data.get('kpis', {}))}

{self._format_trends_section(analytics_data.get('trends', {}))}

{self._format_alerts_section(analytics_data.get('alerts', []))}
"""
        return response.strip()
    
    def _format_kpi_section(self, kpis: Dict[str, Any]) -> str:
        """Format KPI section"""
        
        section = f"""
┌─ {self.emoji_map['sales']} **KEY PERFORMANCE INDICATORS** ──────────────────┐
│                                                            │
│  {self.emoji_map['growth']} **Revenue Growth:** {kpis.get('revenue_growth', '+12.5%'):<20}        │
│  {self.emoji_map['inventory']} **Inventory Turnover:** {kpis.get('inventory_turnover', '8.2x'):<17}        │
│  {self.emoji_map['success']} **Forecast Accuracy:** {kpis.get('forecast_accuracy', '94.2%'):<18}        │
│  {self.emoji_map['user']} **Customer Satisfaction:** {kpis.get('customer_satisfaction', '4.7/5'):<15}        │
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
    def _format_trends_section(self, trends: Dict[str, Any]) -> str:
        """Format trends section"""
        
        section = f"""
┌─ {self.emoji_map['trending']} **MARKET TRENDS** ─────────────────────────────────┐
│                                                            │
│  {self.emoji_map['hot']} **Hot Categories:** {', '.join(trends.get('hot_categories', ['Stationery', 'Electronics']))}     │
│  {self.emoji_map['growth']} **Growing Segments:** {', '.join(trends.get('growing', ['Office', 'Health']))}   │
│  {self.emoji_map['seasonal']} **Seasonal Peaks:** {', '.join(trends.get('seasonal', ['Q4', 'Back-to-School']))}    │
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section
    
    def _format_alerts_section(self, alerts: List[Dict]) -> str:
        """Format alerts section"""
        
        if not alerts:
            return f"""
┌─ {self.emoji_map['success']} **SYSTEM STATUS** ──────────────────────────────────┐
│                                                            │
│  {self.emoji_map['success']} All systems operating normally                        │
│  {self.emoji_map['info']} No critical alerts at this time                        │
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        section = f"""
┌─ {self.emoji_map['alerts']} **ACTIVE ALERTS** ───────────────────────────────────┐
│                                                            │"""
        
        for alert in alerts[:3]:
            severity = alert.get('severity', 'info')
            message = alert.get('message', 'System notification')
            emoji = self.emoji_map.get(severity, self.emoji_map['info'])
            
            section += f"""
│  {emoji} {message:<52} │"""
        
        section += """
│                                                            │
└────────────────────────────────────────────────────────────┘"""
        
        return section