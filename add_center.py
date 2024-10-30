# import os
# import re

# # 指定要处理的目录
# directory = './'  # 请将此处替换为您的实际目录路径

# # 定义正则表达式模式，匹配 ![alt_text](image_path)
# pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

# # 递归遍历目录下的所有 Markdown 文件
# for root, dirs, files in os.walk(directory):
#     for filename in files:
#         if filename.endswith('.md'):
#             filepath = os.path.join(root, filename)
#             with open(filepath, 'r', encoding='utf-8') as file:
#                 content = file.read()

#             # 替换所有匹配的图片格式
#             def replace_image(match):
#                 alt_text = match.group(1)
#                 image_path = match.group(2)
#                 # 获取匹配内容在原始文本中的起始和结束位置
#                 start, end = match.span()
#                 # 检查前面字符是否为换行符
#                 before = content[start-1:start] if start > 0 else ''
#                 # 检查后面字符是否为换行符
#                 after = content[end:end+1] if end < len(content) else ''
#                 # 如果前面不是换行，就在替换内容前添加换行
#                 prefix = '\n' if before != '\n' else ''
#                 # 如果后面不是换行，就在替换内容后添加换行
#                 suffix = '\n' if after != '\n' else ''
#                 new_tag = f'{prefix}<div align="center">\n<img src="{image_path}" width="60%" alt="{alt_text}">\n</div>{suffix}'
#                 return new_tag

#             new_content = pattern.sub(replace_image, content)

#             # 将更新后的内容写回文件
#             with open(filepath, 'w', encoding='utf-8') as file:
#                 file.write(new_content)

#             print(f'已处理文件：{filepath}')

import os
import re

# 指定要处理的目录
directory = '2-llm_compression/'  # 请将此处替换为您的实际目录路径

# 定义正则表达式模式，匹配 <img src="image_path" width="width" alt="alt_text">
pattern = re.compile(
    r'<img\s+[^>]*?src="([^"]+)"[^>]*?width="([^"]+)"[^>]*?alt="([^"]+)"[^>]*/?>',
    re.IGNORECASE
)

# 递归遍历目录下的所有 Markdown 文件
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.md'):
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # 替换所有匹配的图片格式
            def replace_image(match):
                image_path = match.group(1)
                width = match.group(2)
                alt_text = match.group(3)
                # 获取匹配内容在原始文本中的起始和结束位置
                start, end = match.span()
                # 检查前面字符是否为换行符
                before = content[start-1:start] if start > 0 else ''
                # 检查后面字符是否为换行符
                after = content[end:end+1] if end < len(content) else ''
                # 如果前面不是换行，就在替换内容前添加换行
                prefix = '\n' if before != '\n' else ''
                # 如果后面不是换行，就在替换内容后添加换行
                suffix = '\n' if after != '\n' else ''
                new_tag = f'{prefix}<div align="center">\n<img src="{image_path}" width="{width}" alt="{alt_text}">\n</div>{suffix}'
                return new_tag

            new_content = pattern.sub(replace_image, content)

            # 将更新后的内容写回文件
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(new_content)

            print(f'已处理文件：{filepath}')