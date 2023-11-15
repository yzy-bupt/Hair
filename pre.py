from PIL import Image

def resize_image(input_path, output_path, new_size):
    # 打开原始图像
    image = Image.open(input_path)
    
    # 调整图像大小
    resized_image = image.resize(new_size)
    
    # 保存调整后的图像
    resized_image.save(output_path)

# 指定输入和输出文件路径以及新的图像尺寸
input_image_path = 'examples/reference/00058_00.jpg'
output_image_path = 'examples/reference/58.jpg'
new_size = (512, 512)

# 调用函数进行图像尺寸调整
resize_image(input_image_path, output_image_path, new_size)
