from setuptools import find_packages, setup

setup(
    name="milwsi",
    version="0.0.1",
    description="State-of-the-art MIL model for Whole Slide Image",
    license="GPL-3.0 license",
    author="Qilai Zhang",
    author_email="zhang-ql22@mails.tsinghua.edu.cn",
    keywords="computer vision, computational pathology, multiple instance learning, whole slide image",
    url="https://github.com/QilaiZhang/MIL-WSI",
    package_dir={"": "src"},
    packages=find_packages('src'),
    include_package_data=True
)
