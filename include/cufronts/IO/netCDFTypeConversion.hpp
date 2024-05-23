#include <netcdf>

template <typename T1> struct netCDFTypeSelector
{
    using nctype = decltype(netCDF::ncFloat);
};
template <> struct netCDFTypeSelector<int>
{
    using nctype = decltype(netCDF::ncInt);
};
template <> struct netCDFTypeSelector<unsigned int>
{
    using nctype = decltype(netCDF::ncUint);
};
template <> struct netCDFTypeSelector<unsigned char>
{
    using nctype = decltype(netCDF::ncUbyte);
};
template <> struct netCDFTypeSelector<char>
{
    using nctype = decltype(netCDF::ncByte);
};
template <> struct netCDFTypeSelector<double>
{
    using nctype = decltype(netCDF::ncDouble);
};
template <> struct netCDFTypeSelector<short>
{
    using nctype = decltype(netCDF::ncShort);
};
template <> struct netCDFTypeSelector<unsigned short>
{
    using nctype = decltype(netCDF::ncUshort);
};