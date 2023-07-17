#!/usr/bin/env sh
mkdir -p /tmp/legged_control/
file_extension="${1##*.}"

if [ "$file_extension" = "xacro" ]; then
    rosrun xacro xacro "$1" robot_type:="$2" > /tmp/legged_control/"$2".urdf
elif [ "$file_extension" = "urdf" ]; then
    cp -f "$1" /tmp/legged_control/
else
    echo "Unsupported file format. Only xacro and urdf files are supported."
    exit 1
fi
