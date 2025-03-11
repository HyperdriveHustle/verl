#b和mlx5的设备名
ib_devices=""
mlx5_devices=""
mlx_device=""


# 读取/sys/class/net下的ib设备
for net_device in $(ls -l /sys/class/net | grep '^l' | awk '{print $9}' | grep '^ib[0-7]$'); do
    if [ -z "$ib_devices" ];then
       ib_devices="$net_device"
    else
       ib_devices="$ib_devices $net_device"
    fi
done

# 读取/sys/class/infiniband下的mlx5设备
for infiniband_device in $(ls -l /sys/class/infiniband | grep '^l' | awk '{print $9}' | grep -v 'mlx5_bond*'); do
    if [ -z "$mlx5_devices" ];then
        mlx5_devices="$infiniband_device"
    else
        mlx5_devices="$mlx5_devices $infiniband_device"
    fi
done

#echo $ib_devices
#echo $mlx5_devices

oldIFS=$IFS
IFS=' '

# 遍历ib设备，查找对应的mlx5设备
for ib in $ib_devices; do
    ib_path=`readlink -f /sys/class/net/$ib`
    ib_dir=$(echo $ib_path |awk -F'/' '{print $7}')
    for mlx5 in  $mlx5_devices; do
            mlx5_path=$(readlink -f /sys/class/infiniband/$mlx5)
            mlx5_dir=$(echo $mlx5_path |awk -F'/' '{print $7}')
            if [ "$ib_dir" = "$mlx5_dir" ]; then
		    #echo $mlx5
		#echo $ib
		#echo $mlx5_dir, $ib_dir
                #mlx_device=$mlx_device,$mlx5
                if [ -z "$mlx_device" ];then
                   mlx_device="$mlx5"
                else
                   mlx_device="$mlx_device,$mlx5"
                fi
                break
            fi
    done
done


IFS=$oldIFS
export NCCL_IB_HCA=$mlx_device
echo $mlx_device
