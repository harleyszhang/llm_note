import os
import re

# 指定要处理的目录
directory = '1-transformer_model/'  # 请将此处替换为您的实际目录路径

# 定义正则表达式模式，匹配 ![alt_text](image_path)
pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

# 遍历目录下的所有 Markdown 文件
for filename in os.listdir(directory):
    if filename.endswith('.md'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # 替换所有匹配的图片格式
        new_content = pattern.sub(r'<center>\n<img src="\2" width="60%" alt="\1">\n</center>', content)

        # 将更新后的内容写回文件
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(new_content)

        print(f'已处理文件：{filename}')