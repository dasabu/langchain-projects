import streamlit as st
import torch
from ai_services import image_to_text, generate_recipe

torch.classes.__path__ = []

def main():
    st.title('Image to Recipe 🍆')
    
    st.header('Upload an image of ingredients to get a recipe')
    
    uploaded_file = st.file_uploader(
        'Upload an image',
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    
    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption='Uploaded Image',
            use_container_width=True
        )
        file_bytes = uploaded_file.getvalue()
        
        with open(f'./images/{uploaded_file.name}', 'wb') as f:
            f.write(file_bytes)
        
        # Get ingredients caption from image
        with st.spinner('Generating ingredients caption...'):
            ingredients = image_to_text(f'./images/{uploaded_file.name}')

        with st.expander('Ingredients'):
            st.write(ingredients)
        
        # Get recipe from ingredients
        with st.spinner('Generating recipe...'):
            recipe = generate_recipe(ingredients)
        
        with st.expander('Recipe'):
            st.write(recipe)

if __name__ == '__main__':
    main()