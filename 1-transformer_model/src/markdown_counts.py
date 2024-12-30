import os
import re

def count_words_in_markdown(file_path):
    """
    统计单个 Markdown 文件的字数
    :param file_path: Markdown 文件路径
    :return: 文件的字数
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 移除 Markdown 特定格式的内容（如 #、*、链接、代码块等）
        content = re.sub(r'```.*?```', '', content, flags=re.S)  # 移除代码块
        content = re.sub(r'\[.*?\]\(.*?\)', '', content)          # 移除链接
        content = re.sub(r'[#*>\-`]', '', content)               # 移除特殊标记
        content = re.sub(r'\n', ' ', content)                    # 替换换行符为空格
        content = re.sub(r'\s+', ' ', content)                   # 去掉多余的空格
        # 返回分词后字数
        return len(content.split())

def count_words_in_directory(directory_path):
    """
    统计指定目录下所有 Markdown 文件的字数
    :param directory_path: 目录路径
    :return: 各文件字数和总字数
    """
    total_word_count = 0
    markdown_files = []

    # 遍历目录下的 Markdown 文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.md'):  # 仅处理 .md 文件
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)

    # 统计每个文件的字数
    for file_path in markdown_files:
        word_count = count_words_in_markdown(file_path)
        total_word_count += word_count
        print(f"文件: {file_path} - 字数: {word_count}")

    print(f"\n总字数: {total_word_count}")

# 使用示例
directory_path = "/Users/zhg/llm_note/"  # 替换为 Markdown 文件目录的路径
count_words_in_directory(directory_path)