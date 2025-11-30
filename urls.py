from routes import items_bp, sales_bp, user_routes, category_demand_bp, payment_bp

def register_routes(app):
    app.register_blueprint(items_bp, url_prefix='/items')
    app.register_blueprint(sales_bp, url_prefix='/sales')
    app.register_blueprint(user_routes, url_prefix='/users')  
    app.register_blueprint(category_demand_bp, url_prefix="/api")
    app.register_blueprint(payment_bp, url_prefix='/payment')
