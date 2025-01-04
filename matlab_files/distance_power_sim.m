function [power,phase]=distance_power_sim(Power_transmitted,x,y,z,x_int,y_int,z_int,gain_fdrfo,int_gainfo,PLF)

    c=299792458 ; % speed of light
    fo=0.915e9;% first harmonic of signal
    second_harmonic=2*fo;% second harmonic of signal
    interrogator = position(x_int,y_int,z_int);% positon in meters
    node1=position(x,y,z);% position in meters
    Pt=Power_transmitted;
   
   
     gain_fdr2fo=gain_fdrfo; %fdr gain at the moment
     gain_fdrfo=0; %fdr gain 0 at the moment
    
    int_gain2fo=int_gainfo;
     int_gainfo=0;
    load('Conversiongain1.mat')
    

    if node1(1)==0 && node1(2)==0 &&node1(3)<=0
        node_d= sqrt(node1(3)^2)+interrogator(3); % distance between node and int when they are positioned vertically
     elseif node1(1)==0 && node1(2)==0 && node1(3)>0
         node_d=interrogator(3)-sqrt(node1(3)^2);
     else
        %node_d= sqrt((node1(1)^2+node1(2)^2+node1(3)^2)+interrogator(3)^2);
        node_d=sqrt((interrogator(1)-node1(1))^2+(interrogator(2)-node1(2))^2+(interrogator(3)-node1(3))^2);
     end
    %%distance from the interrogator to the node when the inerrogator is fixed
    %%on z axis and node moves about the x axis
  


    path_lossfo=-pathloss(node_d,fo); % node distance in m and frequency in Hz
    path_loss2fo=-pathloss(node_d,second_harmonic);%second_harmonic);% node distance in m and frequency in Hz
    
    Power_at_diode=Pt+int_gainfo+path_lossfo+gain_fdrfo;% diode= receiver?
    if Power_at_diode<-42 || Power_at_diode>0
        error('The power levels are not consistent with conversion gain ');
      
    end
    for j =1:length(Conversiongain1)

        if (-abs(Power_at_diode)+abs(Conversiongain1(j,1)))<=0.5 && (-abs(Power_at_diode)+abs(Conversiongain1(j,1)))>=-0.5
            
            CG=Conversiongain1(j,2);
        end
    end
   
   %CG=0;
   phase1 =((2*pi*fo)/c)*node_d;
   phase2=((2*pi*second_harmonic)/c)*node_d;
   phase=phase1+phase2;
   power=Pt+int_gainfo+path_lossfo+gain_fdrfo+CG+gain_fdr2fo+path_loss2fo+int_gain2fo+PLF; 
   %%link budget to determine the power recieved by the interrogator.

end

