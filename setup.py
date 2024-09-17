from setuptools import setup, find_packages

setup(
    name='ProteinInvBench',              # 包名
    version='0.1.0',                       # 包的版本号
    author='Zhangyang Gao',                    # 作者姓名
    author_email='gaozhangyang@westlake.edu.cn', # 作者邮箱
    description='A brief description of your package',  # 包的简短描述
    long_description=open('README.md').read(),          # 从README.md中读取的长描述
    long_description_content_type='text/markdown',      # 长描述的格式
    url='https://github.com/A4Bio/ProteinInvBench.git',    # 项目的URL（通常是GitHub链接）
    packages=find_packages(),              # 自动查找包
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',   # 许可证类型
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',               # Python版本要求
    install_requires=[                     # 依赖的包列表
    ],
    dependency_links=[
    ],
)
