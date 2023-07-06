#!/bin/sh
for file in *.xz; do
  if [ -e "$file" ]; then
    newname=`echo "$file" | cut -d '_' -f2`
    mv "$file" "$newname"
  fi
done
