#pragma once
struct CsExtent{
    unsigned int width;
    unsigned int height;
    unsigned int samples;
    unsigned int variables;

    CsExtent(unsigned int width, unsigned int height, unsigned int samples, unsigned int variables= 1){
        this->width = width;
        this->height = height;
        this->samples = samples;
        this->variables = variables;
    }
};