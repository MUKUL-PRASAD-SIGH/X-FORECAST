"""
SuperX Supermarket Dataset Generator
Creates realistic 5-year sales data for a supermarket chain
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class SuperXDataGenerator:
    """Generate realistic supermarket sales data for SuperX chain"""
    
    def __init__(self):
        self.products = {
            # Stationery & Office Supplies
            1: {"name": "Apsara Pencil HB", "category": "Stationery", "price": 5.0, "seasonal": False},
            2: {"name": "Apsara Pencil 2B", "category": "Stationery", "price": 5.5, "seasonal": False},
            3: {"name": "Parker Fountain Pen", "category": "Stationery", "price": 250.0, "seasonal": False},
            4: {"name": "Reynolds Ball Pen", "category": "Stationery", "price": 15.0, "seasonal": False},
            5: {"name": "Classmate Notebook A4", "category": "Stationery", "price": 45.0, "seasonal": True},
            6: {"name": "Fevicol Glue Stick", "category": "Stationery", "price": 25.0, "seasonal": True},
            7: {"name": "Stapler Heavy Duty", "category": "Office", "price": 180.0, "seasonal": False},
            8: {"name": "Paper Clips Box", "category": "Office", "price": 35.0, "seasonal": False},
            
            # Groceries & Food
            10: {"name": "Basmati Rice 5kg", "category": "Groceries", "price": 450.0, "seasonal": False},
            11: {"name": "Wheat Flour 10kg", "category": "Groceries", "price": 380.0, "seasonal": False},
            12: {"name": "Toor Dal 1kg", "category": "Groceries", "price": 120.0, "seasonal": False},
            13: {"name": "Sunflower Oil 1L", "category": "Groceries", "price": 140.0, "seasonal": False},
            14: {"name": "Sugar 1kg", "category": "Groceries", "price": 45.0, "seasonal": False},
            15: {"name": "Tea Powder 250g", "category": "Beverages", "price": 85.0, "seasonal": False},
            16: {"name": "Coffee Powder 200g", "category": "Beverages", "price": 180.0, "seasonal": False},
            17: {"name": "Milk 1L Packet", "category": "Dairy", "price": 55.0, "seasonal": False},
            18: {"name": "Bread Loaf", "category": "Bakery", "price": 25.0, "seasonal": False},
            19: {"name": "Biscuits Pack", "category": "Snacks", "price": 35.0, "seasonal": False},
            
            # Personal Care & Hygiene
            20: {"name": "Colgate Toothpaste", "category": "Personal Care", "price": 65.0, "seasonal": False},
            21: {"name": "Shampoo 200ml", "category": "Personal Care", "price": 120.0, "seasonal": False},
            22: {"name": "Soap Bar", "category": "Personal Care", "price": 35.0, "seasonal": False},
            23: {"name": "Face Wash 100ml", "category": "Personal Care", "price": 85.0, "seasonal": False},
            24: {"name": "Deodorant Spray", "category": "Personal Care", "price": 180.0, "seasonal": False},
            
            # Household Items
            25: {"name": "Detergent Powder 1kg", "category": "Household", "price": 95.0, "seasonal": False},
            26: {"name": "Dish Soap 500ml", "category": "Household", "price": 45.0, "seasonal": False},
            27: {"name": "Toilet Paper 4 Roll", "category": "Household", "price": 120.0, "seasonal": False},
            28: {"name": "Floor Cleaner 1L", "category": "Household", "price": 85.0, "seasonal": False},
            29: {"name": "Air Freshener", "category": "Household", "price": 65.0, "seasonal": False},
            
            # Electronics & Accessories
            30: {"name": "Mobile Charger", "category": "Electronics", "price": 350.0, "seasonal": False},
            31: {"name": "Earphones", "category": "Electronics", "price": 450.0, "seasonal": False},
            32: {"name": "Power Bank 10000mAh", "category": "Electronics", "price": 1200.0, "seasonal": False},
            33: {"name": "USB Cable", "category": "Electronics", "price": 180.0, "seasonal": False},
            34: {"name": "Memory Card 32GB", "category": "Electronics", "price": 650.0, "seasonal": False},
            
            # Seasonal Items
            35: {"name": "Umbrella", "category": "Seasonal", "price": 280.0, "seasonal": True},
            36: {"name": "Raincoat", "category": "Seasonal", "price": 450.0, "seasonal": True},
            37: {"name": "Winter Jacket", "category": "Seasonal", "price": 1200.0, "seasonal": True},
            38: {"name": "Summer Hat", "category": "Seasonal", "price": 180.0, "seasonal": True},
            39: {"name": "Sunglasses", "category": "Seasonal", "price": 350.0, "seasonal": True},
            
            # Fruits & Vegetables
            40: {"name": "Apples 1kg", "category": "Fruits", "price": 180.0, "seasonal": True},
            41: {"name": "Bananas 1kg", "category": "Fruits", "price": 60.0, "seasonal": True},
            42: {"name": "Oranges 1kg", "category": "Fruits", "price": 120.0, "seasonal": True},
            43: {"name": "Tomatoes 1kg", "category": "Vegetables", "price": 45.0, "seasonal": True},
            44: {"name": "Onions 1kg", "category": "Vegetables", "price": 35.0, "seasonal": True},
            45: {"name": "Potatoes 1kg", "category": "Vegetables", "price": 25.0, "seasonal": True},
            
            # Medicines & Health
            46: {"name": "Paracetamol Tablets", "category": "Medicine", "price": 25.0, "seasonal": False},
            47: {"name": "Cough Syrup", "category": "Medicine", "price": 85.0, "seasonal": True},
            48: {"name": "Vitamin C Tablets", "category": "Health", "price": 120.0, "seasonal": False},
            49: {"name": "Hand Sanitizer", "category": "Health", "price": 45.0, "seasonal": False},
            50: {"name": "Face Mask Pack", "category": "Health", "price": 35.0, "seasonal": False}
        }
        
        self.stores = {
            1: {"name": "SuperX Mall Road", "location": "Mall Road", "size": "Large"},
            2: {"name": "SuperX City Center", "location": "City Center", "size": "Medium"},
            3: {"name": "SuperX Suburb Plaza", "location": "Suburb", "size": "Small"},
            4: {"name": "SuperX Downtown", "location": "Downtown", "size": "Large"},
            5: {"name": "SuperX Express Highway", "location": "Highway", "size": "Medium"}
        }
        
    def get_product_recommendations(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Get product recommendations for chatbot"""
        products = list(self.products.items())
        
        if category:
            products = [(pid, pinfo) for pid, pinfo in products 
                       if pinfo['category'].lower() == category.lower()]
        
        products.sort(key=lambda x: x[0])
        
        recommendations = []
        for pid, pinfo in products[:limit]:
            recommendations.append({
                'id': pid,
                'name': pinfo['name'],
                'category': pinfo['category'],
                'price': pinfo['price']
            })
        
        return recommendations
    
    def get_categories(self) -> List[str]:
        """Get all product categories"""
        categories = set(product['category'] for product in self.products.values())
        return sorted(list(categories))

# Initialize global dataset
superx_data = SuperXDataGenerator()