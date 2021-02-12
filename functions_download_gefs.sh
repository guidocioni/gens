download_gefs_member()
{
    # Make sure the forecast hour is 2+ digits
    fcstHour=$(printf "%03d" $1)
    for j in {1..30}
    do
        # Make sure the perturbation number is 2 digits
        pert="0${j}"
        pert="${pert: -2}"

        # Download the grib files
        url="https://www.ftp.ncep.noaa.gov/data/nccf/com/gens/prod/gefs."$YEAR""$MONTH""$DATE"/"$RUN"/atmos/pgrb2ap5/gep"$pert".t"$RUN"z.pgrb2a.0p50.f"$fcstHour""
        echo $url
        timeout 40 perl $HOME_SCRIPTS/get_inv.pl "${url}.idx" | \
                grep -E ":(APCP|CSNOW|CRAIN|TMP:2 m above ground|TMP:850 mb|HGT:500 mb|UGRD:10 m above ground|VGRD:10 m above ground|CAPE)" | \
                timeout 40 perl $HOME_SCRIPTS/get_grib.pl "${url}" $GRIBDIR/grib_gefs_"$YEAR""$MONTH""$DATE"_"$RUN"_"$fcstHour"_"$pert"
    done

# Now download the control run
url_control="https://www.ftp.ncep.noaa.gov/data/nccf/com/gens/prod/gefs."$YEAR""$MONTH""$DATE"/"$RUN"/atmos/pgrb2ap5/gec00.t"$RUN"z.pgrb2a.0p50.f"$fcstHour""
echo $url_control
timeout 40 perl $HOME_SCRIPTS/get_inv.pl "${url_control}.idx" | \
        grep -E ":(APCP|CSNOW|CRAIN|TMP:2 m above ground|TMP:850 mb|HGT:500 mb|UGRD:10 m above ground|VGRD:10 m above ground|CAPE)" | \
        timeout 40 perl $HOME_SCRIPTS/get_grib.pl "${url_control}" $GRIBDIR/grib_gefs_"$YEAR""$MONTH""$DATE"_"$RUN"_"$fcstHour"_00
}
export -f download_gefs_member