# Free disk space on Linux
#sudo swapoff /swapfile
#sudo rm -rf /swapfile
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc
#sudo apt-get remove php* ruby-* subversion mongodb-org -yq >/dev/null 2>&1
sudo apt-get autoremove -y >/dev/null 2>&1
sudo apt-get autoclean -y >/dev/null 2>&1
sudo rm -rf /usr/local/lib/android >/dev/null 2>&1
docker rmi $(docker image ls -aq) >/dev/null 2>&1
