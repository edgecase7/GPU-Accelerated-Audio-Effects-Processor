from PIL import Image

def generate_checkerboard(output_path, image_size=512, tile_size=64):
    """
    Generates a checkerboard image and saves it in the P3 (plain text) PPM format.

    Args:
        output_path (str): The path to save the PPM file.
        image_size (int): The width and height of the image in pixels.
        tile_size (int): The size of each square in the checkerboard.
    """
    try:
        # Define the two colors for the checkerboard
        color1 = (255, 255, 255) # White
        color2 = (0, 0, 0)       # Black

        # Create a new blank image
        img = Image.new('RGB', (image_size, image_size))
        pixels = img.load()

        # Fill in the checkerboard pattern
        for y in range(image_size):
            for x in range(image_size):
                if (x // tile_size) % 2 == (y // tile_size) % 2:
                    pixels[x, y] = color1
                else:
                    pixels[x, y] = color2
        
        # --- Save the image in P3 PPM format ---
        with open(output_path, 'w') as f:
            # Write header
            f.write("P3\n")
            f.write(f"{image_size} {image_size}\n")
            f.write("255\n")
            
            # Write pixel data
            for y in range(image_size):
                for x in range(image_size):
                    r, g, b = pixels[x, y]
                    f.write(f"{r} {g} {b} ")
                f.write("\n")
        
        print(f"Successfully generated '{output_path}' ({image_size}x{image_size}).")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The output filename is hardcoded for simplicity in this project
    output_file = "checkerboard.ppm"
    generate_checkerboard(output_file)
