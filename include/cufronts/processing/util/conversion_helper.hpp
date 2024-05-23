#pragma once
#include "cufronts/types/typedefs.hpp"
#include "cufronts/types/math_helper.cuh"
namespace conversion{

    template<class T>
    HOSTDEVICEQUALIFIER
    T dewpoint_temperature(const T temperature, const T relative_humidity, const bool use_arden_buck = false){
        // assume temperature in Kelvin
        // assume relative humidty in percent
        constexpr const T a = 6.122;
        constexpr const T b = 17.67;
        constexpr const T c = 243.5;
        constexpr const T d = 234.5;
        constexpr const T CtoK = 273.15;
        
        const T temperature_celsius = temperature - CtoK;
        if(relative_humidity > 0){
            const T ym = use_arden_buck ? log(relative_humidity/100.0) + ((b-temperature_celsius/d)*(temperature_celsius/(c+temperature_celsius))) : log(relative_humidity/100.0) + ((b)*(temperature_celsius/(c+temperature_celsius)));
            return (c*ym)/(b-ym) + CtoK;
        }
        else{
            // lim ym -> -inf : c * ym / (b + ym)   ignoring potential connections between r and T this converge toward -c. 
            // So in the invalid cases we simply set the limes as the resulting value to prevent invalid values in the calculations
            return -c + CtoK;
        }
        
    }

    template<class T>
    HOSTDEVICEQUALIFIER
    T equivalent_potential_temperature(const T temperature, const T specific_humidity, const T relative_humidity, const T pressure, const bool use_arden_buck = false)
        {
            // assume temperature in Kelvin
            // assume relative humidity in percent
            // assume specific humidity in kg/kg
            constexpr const T k = 287.052874/1005.7;
            constexpr const T CtoK = 273.15;
            
            // dew point temperature
            const T dew_point = dewpoint_temperature(temperature, relative_humidity, use_arden_buck);
            // water vapor pressure (use dew_point as temperature)
            
            const T e = 0.61078*exp(17.27*(dew_point-CtoK)/(dew_point-CtoK+237.3))*10;
            // Attention: This formula returns invalid values for low relative humidities (e.g. 0 or near 0)
            // maybe a slightly better approximation
            // const T e = 0.71121*exp((18.678 - (dew_point-CtoK)/234.5)*((dew_point-CtoK)/(257.14+(dew_point-CtoK))));


            const T TL = 1/ (1/(dew_point-56)+log(temperature/dew_point)/800) + 56;
            const T ThetaDL = temperature*pow(1000.0/(pressure-e), k)* pow(temperature/TL, 0.28*specific_humidity);
            const T Theta_e = ThetaDL*exp(((3036.0/TL)-1.78)*specific_humidity*(1+0.448*specific_humidity));
            return Theta_e;

        }
};
