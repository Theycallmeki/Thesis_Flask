# urls.py
from routes.items import items_bp
# Future blueprints (add later)
# from routes.sales_history import sales_bp
# from routes.users import users_bp


def register_routes(app):
    """
    Attach all route blueprints to the Flask app.
    This keeps app.py clean and modular.
    """
    app.register_blueprint(items_bp)
    # app.register_blueprint(sales_bp)
    # app.register_blueprint(users_bp)
