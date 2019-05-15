# dAUTOMAP

dAUTOMAP: Decomposing AUTOMAP to Achieve Scalability and Enhance Performance

Authors: Jo Schlemper, Ilkay Oksuz, James Clough, Jinming Duan, Andrew King, Julia Schnabel, Joseph Hajnal, Daniel Rueckert

![Fig1](docs/ISMRM2019-000665_Fig1.png)

AUTOMAP is a promising generalized reconstruction approach, however, it is not scalable and hence the practicality is limited. We present a novel way for decomposing the domain transformation, which makes the model scale linearly with the input size. We show the proposed method, termed dAUTOMAP, outperforms AUTOMAP with significantly fewer parameters.

The abstract was presented as an oral presentation during the Scientific Session: "Machine Learning for Image Reconstruction" at ISMRM2019 (https://www.ismrm.org/19/program_files/O49.htm).

This repository contains the code for dAUTOMAP.


Qualitative results for the synthetic experiments with image size 128x128

![Fig1](docs/ISMRM2019-000665_Fig3.png)

Qualitative results for the synthetic experiments with image size 256x256, 2x Cartesian reconstruction

![Fig1](docs/ISMRM2019-000665_Fig4.png)

Quantitative Results:

![Fig1](docs/ISMRM2019-000665_Fig2.png)

MIT License

Copyright (c) [2019] [Jo Schlemper]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
