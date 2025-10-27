from routes import items_bp, sales_bp, user_routes

def register_routes(app):
    app.register_blueprint(items_bp, url_prefix='/items')
    app.register_blueprint(sales_bp, url_prefix='/sales')
    app.register_blueprint(user_routes, url_prefix='/users')  
