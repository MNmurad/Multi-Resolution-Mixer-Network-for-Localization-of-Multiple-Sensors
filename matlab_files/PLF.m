function polLF_db=PLF(Unit_node,Unit_Int)
polLF=(dot(Unit_node,Unit_Int))^2;
polLF_db=20*log10(polLF);
end