function UV=Unit_Vector(theta,phi,r)
x_v = r*sin(theta)*cos(phi);
y_v = r*sin(theta)*sin(phi);
z_v = r*cos(theta);
E_field_node = [x_v y_v z_v];
UV=E_field_node./norm(E_field_node);



end