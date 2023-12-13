state_l = 0.9
state_h = 3

m_left = 0.18
m_right = 0.98
m_bottom = 0.17
m_top = 0.99

plt_h = 2.5
plt_w = 3.5


labels = {}
labels[ 0 ] = { 'name':r'No explicit Cation', 'offset' : (0, 0.005 ) }
labels[ 1 ] = { 'name':r'Na$^{+(1)}$', 'offset' : (0, 0.005 ) }
labels[ 2 ] = { 'name':r'NH$_4^{+(1)}$', 'offset' : ( 0.001, 0 ) }
labels[ 3 ] = { 'name':r'CH$_3$NH$_3^{+(1)}$', 'offset' : (-0.008, -0.001 ) }
labels[ 4 ] = { 'name':r'(CH$_3$)$_4$N$^{+(1)}$', 'offset' : (0, 0.005 ) }



fig = plt.figure(figsize=(plt_w,plt_h))
ax2 = fig.add_subplot(1, 1, 1 )
ax2.plot(df["charge"], df["BE"], "o", markersize = 4)

for i, txt in enumerate(systems_plot):
    x = df["charge"][i] + labels[ i ][ 'offset' ][ 0 ]
    y = df["BE"][i] + labels[ i ][ 'offset' ][ 1 ]
    ax2.text( x, y, labels[ i ][ 'name' ], fontsize = 8 )

xmax = -0.155
xmin = -0.195
ax2.set_xlabel(r'$Surface \ Charge \ \mathrm{Bi}$ $\mathrm{(e)}$', fontsize = 12, labelpad = 0.5)
ax2.set_ylabel( '$\Delta\Omega_{\mathrm{CO_{2}}}$ (eV)', fontsize = 12, labelpad = 2 )
ymin = -0.026
ymax = 0.16
ax2.set_ylim( ymin, ymax)
yticks =  [0, 0.05, 0.1, 0.15 ] 
ax2.set_yticks( yticks )
ax2.set_yticklabels( [ str(x) for x in yticks ] )
ax2.set_xlim( xmin, xmax)
xticks =  [ -0.2, -0.18, -0.16] 
ax2.set_xticks( xticks )
ax2.set_xticklabels( [ str( x ) for x in xticks ] )
ax2.hlines(0, -0.2, xmax, lw = 0.5, color = 'gray' )
ax2.text(-0.175,  0.004 , '$\Phi$ = -1.4 (V vs RHE)', fontsize = 9)
plt.subplots_adjust(left=m_left, right=m_right, top=m_top, bottom=m_bottom, wspace=0.00, hspace= 0.0 )
plt.savefig( 'charge_vs_BE_single_atom.png', dpi = 600 )
plt.show()

