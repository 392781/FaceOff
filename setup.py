from setuptools import setup

setup(
    name='FaceOff',
    version='0.0.0',    
    description='Steps towards physical adversarial attacks on facial recognition',
    url='https://github.com/392781/FaceOff',
    author='Ronald Lencevičius',
    license='GNU General Public License v3 (GPLv3)',
    packages=['FaceOff'],
    install_requires=['torch==1.13.1',
                      'torchvision==0.8.2',
                      'facenet-pytorch==2.5.1',
                      'face_recognition',
                      'face_recognition_models',
                      'cmake',
                      'dlib',
                      'numpy',
                      'Pillow'
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)