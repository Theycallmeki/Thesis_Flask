from routes import items_bp, sales_bp

def register_routes(app):
    app.register_blueprint(items_bp)
    app.register_blueprint(sales_bp)
