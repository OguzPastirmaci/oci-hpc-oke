#!/bin/bash
set -euo pipefail

is_lnet_loaded() {
  grep -q '^lnet ' /proc/modules 2>/dev/null
}

is_ksocklnd_loaded() {
  grep -q '^ksocklnd ' /proc/modules 2>/dev/null
}

is_any_lustre_mounted() {
  awk '$3 == "lustre" { found = 1 } END { exit(found ? 0 : 1) }' /proc/mounts
}

mount_source_for_target() {
  local target="$1"
  awk -v target="$target" '$2 == target { print $1; exit }' /proc/mounts
}

lnet_has_any_tcp_net() {
  awk '
    /^[[:space:]]*-[[:space:]]*net( type)?:[[:space:]]*tcp[[:digit:]]*[[:space:]]*$/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

lnet_has_tcp_iface() {
  local want="$1"
  awk -v want="$want" '
    /^[[:space:]]*-[[:space:]]*net( type)?:/ {
      in_tcp = ($0 ~ /[[:space:]]tcp[[:digit:]]*[[:space:]]*$/)
      next
    }
    in_tcp {
      line = $0
      sub(/^[[:space:]]*/, "", line)
      if (line ~ /^[0-9]+: /) {
        sub(/^[0-9]+: /, "", line)
        sub(/[[:space:]].*$/, "", line)
        if (line == want) {
          found = 1
        }
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

ensure_managed_modprobe_file() {
  local path="$1"
  local line="$2"
  local tmp

  tmp=$(mktemp)
  {
    echo "# Managed by oke-lustre-mount.sh. Do not edit."
    echo "$line"
  } > "$tmp"

  if [ ! -f "$path" ] || ! cmp -s "$tmp" "$path"; then
    mv "$tmp" "$path"
    return 0
  fi

  rm -f "$tmp"
  return 1
}

ensure_fstab_entry() {
  local source="$1"
  local target="$2"
  local desired_line="$source $target lustre defaults,_netdev 0 0"
  local tmp

  if awk -v source="$source" -v target="$target" '
    $1 == source && $2 == target && $3 == "lustre" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' /etc/fstab; then
    echo "Mount entry already exists in /etc/fstab"
    return 0
  fi

  tmp=$(mktemp)
  awk -v source="$source" -v target="$target" '
    !($3 == "lustre" && ($1 == source || $2 == target))
  ' /etc/fstab > "$tmp"
  echo "$desired_line" >> "$tmp"
  mv "$tmp" /etc/fstab
  echo "Added mount entry to /etc/fstab"
}

if [ $# -lt 3 ]; then
  echo "Usage: $0 <lustre_ip> <lustre_fs_name> <mount_point>" >&2
  exit 1
fi

LUSTRE_IP="$1"
LUSTRE_FS_NAME="$2"
MOUNT_POINT="$3"
LUSTRE_SOURCE="${LUSTRE_IP}@tcp:/${LUSTRE_FS_NAME}"

if [ -z "$LUSTRE_IP" ] || [ -z "$LUSTRE_FS_NAME" ] || [ -z "$MOUNT_POINT" ]; then
  echo "Error: lustre_ip, lustre_fs_name, and mount_point must not be empty" >&2
  exit 1
fi

CURRENT_MOUNT_SOURCE="$(mount_source_for_target "$MOUNT_POINT")"
TARGET_ALREADY_MOUNTED=0
if [ "$CURRENT_MOUNT_SOURCE" = "$LUSTRE_SOURCE" ]; then
  TARGET_ALREADY_MOUNTED=1
elif [ -n "$CURRENT_MOUNT_SOURCE" ]; then
  echo "Error: $MOUNT_POINT is already mounted from $CURRENT_MOUNT_SOURCE" >&2
  exit 1
fi

if ! modinfo lnet >/dev/null 2>&1; then
  echo "Lustre client (lnet kernel module) is not available on this node; skipping mount"
  exit 0
fi

LNET_IFACE=$(ip -o route get "$LUSTRE_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')
if [ -z "$LNET_IFACE" ]; then
  LNET_IFACE=$(ip route show default | awk '/default/ {print $5; exit}')
fi
if [ -z "$LNET_IFACE" ]; then
  echo "Error: unable to determine TCP interface for LNet" >&2
  exit 1
fi

case "$LNET_IFACE" in
  rdma*|ib*|mlx*)
    echo "Error: resolved interface '$LNET_IFACE' looks like an RDMA device; refusing to bind LNet TCP to it" >&2
    exit 1
    ;;
esac

LNET_MODPROBE_CONF="/etc/modprobe.d/oke-lustre-lnet.conf"
LNET_MODPROBE_LINE="options lnet networks=\"tcp(${LNET_IFACE})\""
LNET_CONF_UPDATED=0
if ensure_managed_modprobe_file "$LNET_MODPROBE_CONF" "$LNET_MODPROBE_LINE"; then
  echo "Wrote LNet interface preference to $LNET_MODPROBE_CONF"
  LNET_CONF_UPDATED=1
fi

KSOCKLND_MODPROBE_CONF="/etc/modprobe.d/oke-lustre-ksocklnd.conf"
KSOCKLND_MODPROBE_LINE="options ksocklnd skip_mr_route_setup=1"
KSOCKLND_CONF_UPDATED=0
if ensure_managed_modprobe_file "$KSOCKLND_MODPROBE_CONF" "$KSOCKLND_MODPROBE_LINE"; then
  echo "Wrote ksocklnd route-safety setting to $KSOCKLND_MODPROBE_CONF"
  KSOCKLND_CONF_UPDATED=1
fi

if ! is_lnet_loaded; then
  echo "LNet not loaded, loading..."
  modprobe lnet
fi

lnet_output="$(lnetctl net show --verbose 2>&1 || true)"

if printf '%s\n' "$lnet_output" | grep -q "LNet stack down"; then
  lnetctl lnet configure
  lnet_output="$(lnetctl net show --verbose 2>&1 || true)"
fi

NEEDS_TCP_RECONFIGURE=0
if ! printf '%s\n' "$lnet_output" | lnet_has_tcp_iface "$LNET_IFACE"; then
  NEEDS_TCP_RECONFIGURE=1
fi

NEEDS_KSOCKLND_RELOAD=0
if [ "$KSOCKLND_CONF_UPDATED" -eq 1 ] && is_ksocklnd_loaded; then
  NEEDS_KSOCKLND_RELOAD=1
fi

NEEDS_LNET_REFRESH=0
if [ "$NEEDS_TCP_RECONFIGURE" -eq 1 ] || [ "$NEEDS_KSOCKLND_RELOAD" -eq 1 ]; then
  NEEDS_LNET_REFRESH=1
fi

if [ "$TARGET_ALREADY_MOUNTED" -eq 0 ] && [ "$NEEDS_LNET_REFRESH" -eq 1 ]; then
  if is_any_lustre_mounted; then
    echo "Error: another Lustre filesystem is already mounted; refusing to reconfigure the active LNet TCP network automatically" >&2
    exit 1
  fi

  if printf '%s\n' "$lnet_output" | lnet_has_any_tcp_net; then
    echo "Reconfiguring active LNet TCP network to use $LNET_IFACE"
    lnetctl net del --net tcp
  fi

  if [ "$NEEDS_KSOCKLND_RELOAD" -eq 1 ]; then
    echo "Reloading ksocklnd so skip_mr_route_setup=1 takes effect"
    if ! modprobe -r ksocklnd; then
      echo "Error: unable to reload ksocklnd to apply skip_mr_route_setup=1" >&2
      exit 1
    fi
  fi

  lnetctl net add --net tcp --if "$LNET_IFACE" --peer-timeout 180 --peer-credits 120 --credits 1024
  lnet_output="$(lnetctl net show --verbose 2>&1 || true)"
  echo "LNet configured successfully"
fi

if [ "$TARGET_ALREADY_MOUNTED" -eq 1 ] && {
  [ "$LNET_CONF_UPDATED" -eq 1 ] ||
  [ "$KSOCKLND_CONF_UPDATED" -eq 1 ] ||
  [ "$NEEDS_TCP_RECONFIGURE" -eq 1 ]
}; then
  echo "Lustre is already mounted; updated LNet settings will fully apply after the next module reload or reboot."
fi

if [ "$TARGET_ALREADY_MOUNTED" -eq 0 ] && ! printf '%s\n' "$lnet_output" | lnet_has_tcp_iface "$LNET_IFACE"; then
  echo "Error: LNet TCP network is not bound to interface '$LNET_IFACE'" >&2
  exit 1
fi

mkdir -p "$MOUNT_POINT"

if [ "$TARGET_ALREADY_MOUNTED" -eq 1 ]; then
  echo "Lustre volume is already mounted at $MOUNT_POINT"
else
  if mount -t lustre "$LUSTRE_SOURCE" "$MOUNT_POINT"; then
    echo "Successfully mounted $LUSTRE_SOURCE at $MOUNT_POINT"
  else
    echo "Error mounting Lustre volume" >&2
    exit 1
  fi
fi

ensure_fstab_entry "$LUSTRE_SOURCE" "$MOUNT_POINT"
