"""
Shap-E 3D Object Generator Implementation
==========================================

This file contains Python code for implementing OpenAI's Shap-E model
to generate 3D objects from text prompts and images.

Author: Vengelstad
Repository: https://github.com/Vengelstad/Shap_e_Project
Based on: OpenAI Shap-E (https://github.com/openai/shap-e)
"""

import torch
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image


class ShapEGenerator:
    """
    A wrapper class for generating 3D objects using Shap-E.
    Supports both text-to-3D and image-to-3D generation.
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the Shap-E generator with required models.
        
        Args:
            use_gpu (bool): Whether to use GPU if available. Defaults to True.
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the transmitter model (converts latents to 3D)
        print("Loading transmitter model...")
        self.xm = load_model('transmitter', device=self.device)
        
        # Load diffusion configuration
        print("Loading diffusion configuration...")
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        
        # Text and image models will be loaded on demand
        self.text_model = None
        self.image_model = None
    
    def load_text_model(self):
        """Load the text-to-3D model."""
        if self.text_model is None:
            print("Loading text-to-3D model...")
            self.text_model = load_model('text300M', device=self.device)
        return self.text_model
    
    def load_image_model(self):
        """Load the image-to-3D model."""
        if self.image_model is None:
            print("Loading image-to-3D model...")
            self.image_model = load_model('image300M', device=self.device)
        return self.image_model
    
    def generate_from_text(self, prompt, batch_size=1, guidance_scale=15.0, 
                          num_inference_steps=64, output_path=None):
        """
        Generate 3D objects from a text prompt.
        
        Args:
            prompt (str): Text description of the 3D object to generate.
            batch_size (int): Number of 3D objects to generate. Defaults to 1.
            guidance_scale (float): How closely to follow the prompt (higher = more faithful). 
                                   Defaults to 15.0.
            num_inference_steps (int): Number of diffusion steps. Defaults to 64.
            output_path (str): Path to save the 3D mesh (PLY format). If None, doesn't save.
        
        Returns:
            latents: The generated latent representations.
        """
        model = self.load_text_model()
        
        print(f"Generating 3D object from text: '{prompt}'")
        
        # Sample latents using the diffusion model
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_inference_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        print(f"Generated {len(latents)} 3D object(s)")
        
        # Save the first generated object if output path is provided
        if output_path and len(latents) > 0:
            self.save_mesh(latents[0], output_path)
        
        return latents
    
    def generate_from_image(self, image_path, guidance_scale=3.0, 
                           num_inference_steps=64, output_path=None):
        """
        Generate 3D objects from an input image.
        
        Args:
            image_path (str): Path to the input image file.
            guidance_scale (float): How closely to follow the image. Defaults to 3.0.
            num_inference_steps (int): Number of diffusion steps. Defaults to 64.
            output_path (str): Path to save the 3D mesh (PLY format). If None, doesn't save.
        
        Returns:
            latents: The generated latent representations.
        """
        model = self.load_image_model()
        
        print(f"Loading image from: {image_path}")
        image = load_image(image_path, device=self.device)
        
        print("Generating 3D object from image...")
        
        # Sample latents using the diffusion model
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_inference_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        print("3D object generated successfully")
        
        # Save the generated object if output path is provided
        if output_path and len(latents) > 0:
            self.save_mesh(latents[0], output_path)
        
        return latents
    
    def render_latents(self, latents, size=64, rendering_mode='nerf'):
        """
        Render latent representations as images from multiple camera angles.
        
        Args:
            latents: Latent representations to render.
            size (int): Number of camera positions to render. Defaults to 64.
            rendering_mode (str): Rendering mode ('nerf' or 'stf'). Defaults to 'nerf'.
        
        Returns:
            list: List of rendered images for each latent.
        """
        cameras = create_pan_cameras(size, self.device)
        all_images = []
        
        for idx, latent in enumerate(latents):
            print(f"Rendering latent {idx + 1}/{len(latents)}...")
            images = decode_latent_images(
                self.xm, 
                latent, 
                cameras, 
                rendering_mode=rendering_mode
            )
            all_images.append(images)
        
        return all_images
    
    def save_mesh(self, latent, output_path):
        """
        Save a latent representation as a 3D mesh file.
        
        Args:
            latent: Latent representation to convert to mesh.
            output_path (str): Path where the mesh file will be saved (PLY format).
        """
        print(f"Saving mesh to: {output_path}")
        
        # Decode latent to mesh and save
        mesh = decode_latent_mesh(self.xm, latent).tri_mesh()
        
        with open(output_path, 'wb') as f:
            mesh.write_ply(f)
        
        print(f"Mesh saved successfully!")
    
    def save_as_gif(self, images, output_path):
        """
        Save rendered images as an animated GIF using PIL.
        
        Args:
            images: List of rendered images (numpy arrays).
            output_path (str): Path where the GIF will be saved.
        """
        print(f"Saving animation to: {output_path}")
        
        pil_images = [Image.fromarray(img) for img in images]
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=50,
            loop=0
        )
        print(f"GIF saved successfully!")


def example_text_to_3d():
    """
    Example: Generate a 3D object from a text prompt.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Text-to-3D Generation")
    print("="*60 + "\n")
    
    # Initialize the generator
    generator = ShapEGenerator(use_gpu=True)
    
    # Define prompts to try
    prompts = [
        "a red vintage car",
        "a wooden chair",
        "a coffee mug with steam"
    ]
    
    for idx, prompt in enumerate(prompts):
        output_file = f"output_text_{idx+1}.ply"
        
        # Generate 3D object
        latents = generator.generate_from_text(
            prompt=prompt,
            batch_size=1,
            guidance_scale=15.0,
            output_path=output_file
        )
        
        # Render the object
        images = generator.render_latents(latents, size=32)
        
        # Save as GIF
        gif_file = f"output_text_{idx+1}.gif"
        generator.save_as_gif(images[0], gif_file)
        
        print(f"\nGenerated 3D object saved to: {output_file}")
        print(f"Animation saved to: {gif_file}\n")


def example_image_to_3d():
    """
    Example: Generate a 3D object from an input image.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Image-to-3D Generation")
    print("="*60 + "\n")
    
    # Initialize the generator
    generator = ShapEGenerator(use_gpu=True)
    
    # Path to input image (replace with your own image)
    input_image = "input_image.jpg"
    output_file = "output_from_image.ply"
    
    # Generate 3D object from image
    latents = generator.generate_from_image(
        image_path=input_image,
        guidance_scale=3.0,
        output_path=output_file
    )
    
    # Render the object
    images = generator.render_latents(latents, size=32)
    
    # Save as GIF
    gif_file = "output_from_image.gif"
    generator.save_as_gif(images[0], gif_file)
    
    print(f"\nGenerated 3D object saved to: {output_file}")
    print(f"Animation saved to: {gif_file}\n")


def example_batch_generation():
    """
    Example: Generate multiple 3D objects from a single prompt.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Generation")
    print("="*60 + "\n")
    
    # Initialize the generator
    generator = ShapEGenerator(use_gpu=True)
    
    prompt = "a colorful mushroom"
    batch_size = 4
    
    # Generate multiple variations
    latents = generator.generate_from_text(
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=15.0
    )
    
    # Save each generated object
    for idx, latent in enumerate(latents):
        output_file = f"batch_output_{idx+1}.ply"
        generator.save_mesh(latent, output_file)
        print(f"Saved variation {idx+1} to: {output_file}")


def main():
    """
    Main function demonstrating various Shap-E capabilities.
    """
    print("\n" + "="*60)
    print("Shap-E 3D Object Generator - Implementation Examples")
    print("="*60 + "\n")
    
    print("This script demonstrates how to use OpenAI's Shap-E model")
    print("to generate 3D objects from text prompts and images.")
    print("\nNote: Make sure you have installed the shap-e package:")
    print("  pip install -e git+https://github.com/openai/shap-e.git#egg=shap-e")
    print("\n")
    
    # Run examples
    try:
        example_text_to_3d()
    except Exception as e:
        print(f"Error in text-to-3D example: {e}")
    
    try:
        example_image_to_3d()
    except Exception as e:
        print(f"Error in image-to-3D example: {e}")
    
    try:
        example_batch_generation()
    except Exception as e:
        print(f"Error in batch generation example: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
