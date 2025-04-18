from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError
from modules.models import User

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')
            
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class ProductForm(FlaskForm):
    product_id = StringField('Product ID', validators=[DataRequired()])
    name = StringField('Product Name', validators=[DataRequired()])
    price = FloatField('Price', validators=[DataRequired()])
    manufacturer = StringField('Manufacturer', validators=[Optional()])
    category = StringField('Category', validators=[Optional()])
    subcategory = StringField('Subcategory', validators=[Optional()])
    material = StringField('Material', validators=[Optional()])
    dimensions = StringField('Dimensions', validators=[Optional()])
    warranty = IntegerField('Warranty (Years)', validators=[Optional()])
    description = TextAreaField('Description', validators=[Optional()])
    stock_quantity = IntegerField('Stock Quantity', validators=[Optional()])
    submit = SubmitField('Save Product')

class CustomerForm(FlaskForm):
    name = StringField('Customer Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone', validators=[Optional()])
    address = TextAreaField('Address', validators=[Optional()])
    submit = SubmitField('Save Customer')

class SearchForm(FlaskForm):
    query = StringField('Search Query', validators=[DataRequired()])
    submit = SubmitField('Search')

class ApiKeyForm(FlaskForm):
    key_name = StringField('Key Name', validators=[DataRequired()])
    submit = SubmitField('Generate API Key')