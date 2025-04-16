# Door Inventory Management System

A Flask-based web application for managing commercial door products with AI-powered search and invoice generation capabilities.

## Features

- **CSV Data Import**: Upload and analyze CSV files containing door product data
- **AI-Powered Search**: Use natural language to search and analyze your inventory
- **Invoice Generation**: Create professional invoices for selected products
- **Shopping Cart**: Add multiple items to your cart before generating an invoice

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI model access

### Installation

1. Clone the repository or download the source code
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up environment variables:
   ```
   # Create a .env file in the project root
   GOOGLE_API_KEY=your_google_api_key
   SECRET_KEY=your_secret_key
   ```

### Running the Application

For development:
```
flask run --debug
```

For production:
```
flask run
```

## Usage

1. **Upload Data**: Start by uploading a CSV file with your door product data
2. **Enter API Key**: Provide your Google API key (only needed once)
3. **Query Products**: Use natural language to search your inventory
4. **Add to Cart**: Add products to your shopping cart
5. **Generate Invoice**: Create a professional PDF invoice for your order

## Troubleshooting

- If the AI search isn't working, verify your Google API key is valid
- For CSV import issues, ensure your file has the required columns (Product ID, Product Name, Price)
- If invoice generation fails, check that you have at least one item in your cart