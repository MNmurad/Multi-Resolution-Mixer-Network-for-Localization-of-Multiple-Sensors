%% Figuring out the power recieved and phase for each interrogator and node 
function Power=power_calculated(node,int)
    % input: node and int object
    % output: Power for the particulat position of the node.
    
    phasor_voltages=zeros();
    Pt=35; % Power Transmitted in dbm 
    % Pt_list = [35, 30, 40];
    index = 1;
    N_Int = 1; %% Number of interrogators
    
    % % % % % % % % % % % % % % % % % 
    % N_nodes = length(node);%% Number of nodes
    % if node(1).x==node(2).x && node(1).y==node(2).y && node(1).z==node(2).z
    %     N_nodes=1;
    % end
    % % % % % % % % % % % % % % 
    N_nodes = 1;%% Number of nodes
    % % % % % % % % 



    for j=1:N_Int
        % Pt = Pt_list(j);
        int_pos=position(int(j).x,int(j).y,int(j).z); %% getting the node position
        U_int=Unit_Vector(int(j).theta,int(j).phi,int(j).r);
        for i=1:N_nodes 

            node_pos=position(node(i).x,node(i).y,node(i).z); %% getting the node position

            ref_int_pos=position(int_pos(1)-node_pos(1),int_pos(2)-node_pos(2),int_pos(3)-node_pos(3));%% reference interrogator position
            [azimuth,elevation,r] = cart2sph(ref_int_pos(1),ref_int_pos(2),ref_int_pos(3));
            ref_theta=pi/2-elevation; %getting the angular direction to see the gain due to the pattern
            ref_phi=azimuth;

            D=4*cos(ref_theta);%% Directivity considered from the antenna pattern
            D_db=10*log10(D); %% finding the gain in db
            % disp(D_db)
            Un=Unit_Vector(node(i).theta,node(i).phi,node(i).r);%% Unit_vector for nodes
            Poldb=PLF(Un,U_int);%% Loss due to the polarization of the node and interrogator

            int_pos_new=position(0,0,0); %% Looking at the interrogator now as the origin
            ref_node_pos=position(int_pos(1)-node_pos(1),int_pos(2)-node_pos(2),int_pos(3)-node_pos(3)); %% Observation of node by interrogator

            [azimuth_int,elevation_int,r_int] = cart2sph(ref_node_pos(1),ref_node_pos(2),ref_node_pos(3));
            ref_theta_int=pi/2-elevation_int;
            ref_phi_int=azimuth;

            D_int=4*cos(ref_theta_int); %% taken from the radiation pattern of the interrogator
            D_db_int=10*log10(D_int); %% gain for the interrogator
            % disp(D_db)
            [Pr_one,phase]=distance_power_sim(Pt,node_pos,node_pos(2),node_pos(3),int_pos(1),int_pos(2),int_pos(3),D_db,D_db_int,Poldb);
            impedance_antenna=50;
            Power_observed_watts=10^((Pr_one-30)/10); % conversion to watts
            voltage_2=sqrt(2*impedance_antenna*Power_observed_watts);% Finding the incidence voltage magnitude

            phasor_voltages(index)=voltage_2*exp(1i*phase); %adding the phase
            index=index+1;
        end
    end

    Total_Voltage=sum(phasor_voltages);
    Total_Power_dbm=10*log10((1/2*abs(Total_Voltage)^2)/50)+30;
    Power=Total_Power_dbm;
end