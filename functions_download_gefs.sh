download_gefs_member()
{
    # Make sure the forecast hour is 2+ digits
    fcstHour=$(printf "%03d" $1)
    pert=$(printf "%02d" $2)
    if [ $2 -ge 1 ] && [ $2 -le 30 ]
    then
        pert_string="gep${pert}"
    else # control run
        pert_string="gec00"
    fi

    # Download the grib files
    url="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs."$YEAR""$MONTH""$DATE"/"$RUN"/atmos/pgrb2ap5/"$pert_string".t"$RUN"z.pgrb2a.0p50.f"$fcstHour""
    echo $url
    perl $HOME_FOLDER/get_inv.pl "${url}.idx" | \
                grep -E ":(APCP|CSNOW|CRAIN|TMP:2 m above ground|TMP:850 mb|HGT:500 mb|UGRD:10 m above ground|VGRD:10 m above ground|CAPE)" | \
                perl $HOME_FOLDER/get_grib.pl "${url}" $MODEL_DATA_FOLDER/grib_gefs_"$YEAR""$MONTH""$DATE"_"$RUN"_"$fcstHour"_"$pert"
}
export -f download_gefs_member
#
merge_gefs_member()
{
    pert=$(printf "%02d" $1)
    echo "Merging perturbation ${pert}"

    # We need sellonlatbox to shift the grid from 0,360 to -180, 180. Somehow it is
    # easier to do it now that afterwars in Python

    cdo -f nc copy -sellonlatbox,-180,180,-90,90 -mergetime \
                    $MODEL_DATA_FOLDER"grib_gefs_"$YEAR$MONTH$DATE"_"$RUN"_*_${pert}" \
                    $MODEL_DATA_FOLDER"grib_gefs_"$YEAR$MONTH$DATE"_"$RUN"_${pert}.nc"
    rm $MODEL_DATA_FOLDER"grib_gefs_"$YEAR$MONTH$DATE"_"$RUN"_*_${pert}"
}
export -f merge_gefs_member