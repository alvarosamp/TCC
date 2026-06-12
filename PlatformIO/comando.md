usbipd attach --wsl --busid 1-3 -> Rodar no powershell adm p wsl ver a porta

netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=172.19.110.253

netsh advfirewall firewall add rule name="WSL2 Port 8000" dir=in action=allow protocol=TCP 
usbipd attach --wsl --busid 1-1