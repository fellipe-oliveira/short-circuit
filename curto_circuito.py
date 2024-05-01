## Atenção: os motores estão sendo incluídos apenas na sequência positiva.
## Caso sejam declarados motores, não calcular curto-circuito desequilibrado.

## Packages

import pandas as pd
import numpy as np
from sys import argv
from unidecode import unidecode

## Entrada de dados

data = pd.read_excel(argv[1], sheet_name=None)

execucao = data['Execução']
barras = data['Barras']
cargas_estaticas = data['Cargas estáticas']
motores = data['Motores']
linhas = data['Linhas']
transformadores = data['Transformadores']
geradores = data['Geradores']
tts = data['TTs']

f = execucao['Frequência (Hz)'].iloc[0]
sb = execucao['Potência base (MVA)'].iloc[0]
tipo_de_falta = unidecode(execucao['Tipo de falta'].iloc[0].lower())
v_pre_falta = execucao['Tensão pré-falta'].iloc[0]

################################################################################
if len(motores)>0 and tipo_de_falta != 'trifasica':
    raise ValueError('Retire os motores do arquivo de entrada para cálculos de falta desequilibrada.')
################################################################################

# Aplicação de valores default

default = {
    'Execução': {'Potência base (MVA)': 1, 'Tipo de falta': 'Trifásica'},
    'Barras': {'Em serviço': True},
    'Cargas estáticas': {'P (MW)': 0, 'Q (Mvar)': 0, '% P cte': 100, '% I cte': 0, '% Z cte': 0, 'Em serviço':True},
    'Motores': {'Pmec (MW)': 0, 'Eficiência (%)': 90, 'fp':0.9, 'I_rotor_bloq (pu)':7, 'Em serviço':True},
    'Linhas': {'R (ohms/km)': 0, 'C (nF/km)': 0, 'Comprimento (km)':1, 'Condutores paralelos':1, 'Em serviço':True},
    'Transformadores': {'Tap (De)': 1, 'Fase Tap (De)': 0, 'Tap (Para)': 1, 'Fase Tap (Para)': 0,
                        'Em serviço':True, 'R_at_de (ohms)': 0, 'X_at_de (ohms)':0,
                        'R_at_para (ohms)':0, 'X_at_para (ohms)':0, 'Conexao_de':'D', 'Conexao_para':'Yt'},
    'Geradores': {'R_at (ohms)': 0, 'X_at (ohms)': 0, 'Conexão':'Yt', 'Em serviço':True},
    'TTs': {'R0 (ohms)':0, 'X0 (ohms)':0}
}
for df, key in zip([barras, cargas_estaticas, motores, linhas, transformadores, geradores, tts],
                   ['Barras', 'Cargas estáticas', 'Motores', 'Linhas', 'Transformadores', 'Geradores', 'TTs']):
    df.fillna(default[key], inplace=True)

# Indexes dos DataFrames que constituem a rede

barras.set_index('ID', inplace=True)

# Retirada dos equipamentos que não estão em serviço

for df in (barras, cargas_estaticas, motores, linhas, transformadores, geradores, tts):
    df.drop(df[df['Em serviço']==False].index, inplace=True)

# Cálculo de admitâncias com correções de base
# As impedâncias de aterramento podem ser nulas. Assim, o cálculo da admitância de aterramento
# não pode ser feito pelo computador, pois o cálculo pode envolver uma divisão por zero.

linhas['R'] = linhas['R (ohms/km)']*linhas['Comprimento (km)']*(sb/(linhas['Vn (kV)']**2))/linhas['Condutores paralelos']
linhas['X'] = linhas['X (ohms/km)']*linhas['Comprimento (km)']*(sb/(linhas['Vn (kV)']**2))/linhas['Condutores paralelos']
linhas['R0'] = linhas['R0 (ohms/km)']*linhas['Comprimento (km)']*(sb/(linhas['Vn (kV)']**2))/linhas['Condutores paralelos']
linhas['X0'] = linhas['X0 (ohms/km)']*linhas['Comprimento (km)']*(sb/(linhas['Vn (kV)']**2))/linhas['Condutores paralelos']

linhas['Y0'] = 1/(linhas['R0'] + 1j*linhas['X0'])
linhas['Y1'] = 1/(linhas['R'] + 1j*linhas['X'])
linhas['Y2'] = 1/(linhas['R'] + 1j*linhas['X'])
linhas['Yshunt'] = 2*np.pi*f*linhas['C (nF/km)']*linhas['Condutores paralelos']*1E-9*linhas['Vn (kV)']**2/sb
linhas['Yshunt0'] = 2*np.pi*f*linhas['C0 (nF/km)']*linhas['Condutores paralelos']*1E-9*linhas['Vn (kV)']**2/sb

for idx, barra, X0, X1, X2, vn_kv, sn in zip(geradores.index, geradores['Barra'], geradores['X0 (%)'], geradores['X1 (%)'], geradores['X2 (%)'], geradores['Vn (kV)'], geradores['Sn (MVA)']):
    vn_barra = barras.at[barra, 'Vn (kV)']
    geradores.loc[idx, 'Z0'] = 1j*X0*(vn_kv/vn_barra)**2*sb/(100*sn)
    geradores.loc[idx, 'Z1'] = 1j*X1*(vn_kv/vn_barra)**2*sb/(100*sn)
    geradores.loc[idx, 'Z2'] = 1j*X2*(vn_kv/vn_barra)**2*sb/(100*sn)
    geradores.loc[idx, 'Y0'] = 100/(1j*X0*(vn_kv/vn_barra)**2*sb/sn)
    geradores.loc[idx, 'Y1'] = 100/(1j*X1*(vn_kv/vn_barra)**2*sb/sn)
    geradores.loc[idx, 'Y2'] = 100/(1j*X2*(vn_kv/vn_barra)**2*sb/sn)
    geradores.loc[idx, 'Zat'] = (geradores.loc[idx, 'R_at (ohms)']+1j*geradores.loc[idx, 'X_at (ohms)'])*sb/vn_barra**2

transformadores['Z'] = transformadores['Z (%)']*sb/(100*transformadores['Sn (MVA)'])
transformadores['R'] = (transformadores['Z']**2/(1+(transformadores['X/R'])**2))**0.5
transformadores['X'] = (transformadores['Z']**2-transformadores['R']**2)**0.5
transformadores['Z0'] = transformadores['R'] + 1j*transformadores['X']
transformadores['Z1'] = transformadores['R'] + 1j*transformadores['X']
transformadores['Z2'] = transformadores['R'] + 1j*transformadores['X']
transformadores['Y0'] = 1/(transformadores['R'] + 1j*transformadores['X'])
transformadores['Y1'] = 1/(transformadores['R'] + 1j*transformadores['X'])
transformadores['Y2'] = 1/(transformadores['R'] + 1j*transformadores['X'])
transformadores['Conexao'] = transformadores['Conexao_de'] + '-' + transformadores['Conexao_para']

motores['Sn (MVA)'] = motores['Pmec (MW)']/(motores['Eficiência (%)']*motores['fp']/100)
motores['Zm'] = motores['Vn (kV)']**2/(motores['I_rotor_bloq (pu)']*motores['Sn (MVA)']*3**0.5)
for idx, barra in zip(motores.index, motores['Barra']):
    motores['Y1'] = barras.at[barra, 'Vn (kV)']**2/(motores['Zm']*sb)

for idx, barra in zip(tts.index, tts['Barra']):
    tts.loc[idx, 'Z0'] = (tts.loc[idx, 'R0 (ohms)'] + 1j*tts[idx, 'X0 (ohms)'])*sb/barras.at[barra, 'Vn (kV)']**2
    tts.loc[idx, 'Y0'] = 1/tts.loc[idx, 'Z0']

for idx, de, para, r_at_prim, x_at_prim, r_at_sec, x_at_sec in zip(transformadores.index, transformadores['De'], transformadores['Para'],
                                                                transformadores['R_at_de (ohms)'], transformadores['X_at_de (ohms)'],
                                                                transformadores['R_at_para (ohms)'], transformadores['X_at_para (ohms)']):
    transformadores['Zat1'] = (r_at_prim + 1j*x_at_prim)*sb/barras.at[de, 'Vn (kV)']**2
    transformadores['Zat2'] = (r_at_sec + 1j*x_at_sec)*sb/barras.at[para, 'Vn (kV)']**2

a = np.exp(1j*2*np.pi/3)

T = np.array([[1,1,1],[1,a**2,a],[1,a,a**2]])
invT = np.array([[1,1,1],[1,a,a**2],[1,a**2,a]])

## Montagem das matrizes de sequência positiva

Ybus1 = pd.DataFrame(0, index=barras.index, columns=barras.index, dtype=complex)

if len(linhas) > 0:
    for de, para, Y1 in zip(linhas['De'], linhas['Para'], linhas['Y1']):
        Ybus1.at[de, de] += Y1
        Ybus1.at[de, para] += -Y1
        Ybus1.at[para, de] += -Y1
        Ybus1.at[para, para] += Y1

if len(transformadores) > 0:
    for de, para, Y1 in zip(transformadores['De'], transformadores['Para'], transformadores['Y1']):
        Ybus1.at[de, de] += Y1
        Ybus1.at[de, para] += -Y1
        Ybus1.at[para, de] += -Y1
        Ybus1.at[para, para] += Y1

if len(geradores) > 0:
    for barra, Y1 in zip(geradores['Barra'], geradores['Y1']):
        Ybus1.at[barra, barra] += Y1

if len(motores) > 0:
    for barra, Y1 in zip(motores['Barra'], motores['Y1']):
        Ybus1.at[barra, barra] += Y1

## Montagem das matrizes de sequência negativa

Ybus2 = pd.DataFrame(0, index=barras.index, columns=barras.index, dtype=complex)

if len(linhas) > 0:
    for de, para, Y2 in zip(linhas['De'], linhas['Para'], linhas['Y2']):
        Ybus2.at[de, de] += Y2
        Ybus2.at[de, para] += -Y2
        Ybus2.at[para, de] += -Y2
        Ybus2.at[para, para] += Y2

if len(transformadores) > 0:
    for de, para, Y2 in zip(transformadores['De'], transformadores['Para'], transformadores['Y2']):
        Ybus2.at[de, de] += Y2
        Ybus2.at[de, para] += -Y2
        Ybus2.at[para, de] += -Y2
        Ybus2.at[para, para] += Y2

if len(geradores) > 0:
    for barra, Y2 in zip(geradores['Barra'], geradores['Y2']):
        Ybus2.at[barra, barra] += Y2

## Montagem das matrizes de sequência 0

Ybus0 = pd.DataFrame(0, index=barras.index, columns=barras.index, dtype=complex)

if len(linhas) > 0:
    for de, para, Y0 in zip(linhas['De'], linhas['Para'], linhas['Y0']):
        Ybus0.at[de, de] += Y0
        Ybus0.at[de, para] += -Y0
        Ybus0.at[para, de] += -Y0
        Ybus0.at[para, para] += Y0

if len(transformadores) > 0:
    for idx in transformadores.index:
        conexao = transformadores.at[idx,'Conexao']
        if conexao in ['D-D','D-Y','Y-D','Y-Y','Yt-Y','Y-Yt']:
            continue
        elif conexao == 'Yt-D':
            de = transformadores.at[idx,'De']
            Z0 = transformadores.at[idx,'Z0']
            Zat1 = transformadores.at[idx,'Zat1']
            Y0 = 1/(Z0+3*Zat1)
            
            Ybus0.at[de, de] += Y0
        elif conexao == 'D-Yt':
            para = transformadores.at[idx, 'Para']
            Z0 = transformadores.at[idx, 'Z0']
            Zat2 = transformadores.at[idx, 'Zat2']
            Y0 = 1/(Z0+3*Zat2)
            
            Ybus0.at[para, para] += Y0
        elif conexao == 'Yt-Yt':
            de = transformadores.at[idx, 'De']
            para = transformadores.at[idx, 'Para']
            Z0 = transformadores.at[idx, 'Z0']
            Zat1 = transformadores.at[idx, 'Zat1']
            Zat2 = transformadores.at[idx, 'Zat2']
            Y0 = 1/(Z0+3*Zat1+3*Zat2) 

            Ybus0.at[de, de] += Y0
            Ybus0.at[de, para] += -Y0
            Ybus0.at[para, de] += -Y0
            Ybus0.at[para, para] += Y0
        else:
            raise ValueError('Preenchimento incorreto da conexão do transformador ' + idx + '.')

if len(geradores) > 0:        
    for idx in geradores.index:
        conexao = geradores.at[idx, 'Conexão']
        if conexao in ['Y','D']:
            continue
        elif conexao == 'Yt':
            barra = geradores.at[idx, 'Barra']
            Z0 = geradores.at[idx, 'Z0']
            Zat = geradores.at[idx, 'Zat']
            Y0 = 1/(Z0+3*Zat)
            
            Ybus0.at[barra, barra] += Y0
        else:
            print('Preenchimento incorreto da conexão do gerador ' + idx + '.')

if len(tts) > 0:
    for barra, Y0 in zip(tts['Barra'], tts['Y0']):
        Ybus0.at[barra, barra] += Y0

## Obtenção das matrizes impedância

Zbus0 = pd.DataFrame(np.linalg.inv(Ybus0), index=barras.index, columns=barras.index, dtype=complex)
Zbus1 = pd.DataFrame(np.linalg.inv(Ybus1), index=barras.index, columns=barras.index, dtype=complex)
Zbus2 = pd.DataFrame(np.linalg.inv(Ybus2), index=barras.index, columns=barras.index, dtype=complex)

## Cálculo da corrente de falta em todas as barras

for barra_em_curto in barras.index:
    Z0th = Zbus0.at[barra_em_curto, barra_em_curto]
    Z1th = Zbus1.at[barra_em_curto, barra_em_curto]
    Z2th = Zbus2.at[barra_em_curto, barra_em_curto]

    I0 = pd.Series(0, index=barras.index, dtype=complex)
    I1 = pd.Series(0, index=barras.index, dtype=complex)
    I2 = pd.Series(0, index=barras.index, dtype=complex)

    if tipo_de_falta == 'fase-terra':
        Ia0 = v_pre_falta/(Z0th+Z1th+Z2th)
        I0[barra_em_curto] = -Ia0
        I1[barra_em_curto] = -Ia0
        I2[barra_em_curto] = -Ia0
        If_pu = 3*Ia0
    elif tipo_de_falta == 'fase-fase':
        Ia1 = v_pre_falta/(Z1th+Z2th)
        I1[barra_em_curto] = -Ia1
        I2[barra_em_curto] = Ia1
        If_pu = a**2*(-Ia1)+a*Ia1
    elif tipo_de_falta == 'trifasica':
        If_pu = v_pre_falta/Z1th
        I1[barra_em_curto] = -If_pu
    elif tipo_de_falta == 'fase-fase-terra':
        Ia1 = v_pre_falta/(Z1th+Z2th*Z0th/(Z2th+Z0th))
        Ia0 = -Z2th/(Z1th*Z2th+Z1th*Z0th+Z2th*Z0th)
        Ia2 = -Z0th/(Z1th*Z2th+Z1th*Z0th+Z2th*Z0th)
        If_pu = 3*Ia0
        I0[barra_em_curto] = -Ia0
        I1[barra_em_curto] = -Ia1
        I2[barra_em_curto] = -Ia2
    else:
        raise ValueError('Preenchimento incorreto do tipo de curto.')

    If = If_pu * sb/(3**0.5*barras.at[barra_em_curto, 'Vn (kV)'])
    barras.loc[barra_em_curto, 'Corrente de curto-circuito (kA)'] = abs(If)

print(f'\nFalta {execucao["Tipo de falta"].iloc[0]}\n')
print(barras)

'''
## Cálculo das tensões 012 nas barras

E = np.ones((nBarras,1),dtype=complex)

V0 = Zbus0.dot(I0)
V1 = Zbus1.dot(I1)+E
V2 = Zbus2.dot(I2)

# Aplicando defasagem nas sequências positiva e negativa

barrasAnalisadas = []
proximasParaAnalisar = [barra_em_curto]
barras.at[barra_em_curto,'Ref'] = 0

# Conexões da barra em curto
# Todas as barras do sistema se conectam com outras barras via linha ou via transformador.

for barra in proximasParaAnalisar:
    conexoesPorLinhasDe = linhas[linhas['De']==barra]
    conexoesPorLinhasPara = linhas[linhas['Para']==barra]
    conexoesPorTransformadoresDe = transformadores[transformadores['De']==barra]
    conexoesPorTransformadoresPara = transformadores[transformadores['Para']==barra]

    for ID in conexoesPorLinhasDe.index:
        de = linhas.at[ID,'De']
        para = linhas.at[ID,'Para']
        barras.at[para,'Ref'] = barras.at[de,'Ref']
        if para in barrasAnalisadas:
            continue
        else:
            proximasParaAnalisar.append(para)
    for ID in conexoesPorLinhasPara.index:
        de = linhas.at[ID,'De']
        para = linhas.at[ID,'Para']
        barras.at[de,'Ref'] = barras.at[para,'Ref']
        if de in barrasAnalisadas:
            continue
        else:
            proximasParaAnalisar.append(de)
    for ID in conexoesPorTransformadoresDe.index:
        de = transformadores.at[ID,'De']
        para = transformadores.at[ID,'Para']
        fase_alfa = transformadores.at[ID,'Fase_Alfa']
        fase_beta = transformadores.at[ID,'Fase_Beta']
        barras.at[para,'Ref'] = barras.at[de,'Ref'] + fase_beta - fase_alfa
        if para in barrasAnalisadas:
            continue
        else:
            proximasParaAnalisar.append(para)
    for ID in conexoesPorTransformadoresPara.index:
        de = transformadores.at[ID,'De']
        para = transformadores.at[ID,'Para']
        fase_alfa = transformadores.at[ID,'Fase_Alfa']
        fase_beta = transformadores.at[ID,'Fase_Beta']
        barras.at[de,'Ref'] = barras.at[para,'Ref'] + fase_alfa - fase_beta
        if de in barrasAnalisadas:
            continue
        else:
            proximasParaAnalisar.append(de)
    barrasAnalisadas.append(barra)

# Aplicando defasagem nas tensões

defasagem1 = np.array([[np.exp(1j*barras.at[barra,'Ref']*np.pi/180)] for barra in barras.index])
defasagem2 = np.array([[np.exp(-1j*barras.at[barra,'Ref']*np.pi/180)] for barra in barras.index])
V1 = V1*defasagem1
V2 = V2*defasagem2

# Cálculo das tensões ABC nas barras
    
aux = []
for barra in barras.index:
    V012 = np.array([[V0[barra][0]],[V1[barra][0]],[V2[barra][0]]])
    Vabc = T.dot(V012)
    aux.append([barra,Vabc[0][0],Vabc[1][0],Vabc[2][0]])
Tensoes = pd.DataFrame(data=aux,columns=['Barra','Va','Vb','Vc'])

## Cálculo das contribuições

aux = []
for ID in transformadores.index:
    de = transformadores.at[ID,'De']
    para = transformadores.at[ID,'Para']
    
    Ynodal1 = matrizesTransformadores1[ID]
    Vnodal1 = np.array([[V1[de][0]],[V1[para][0]]])
    Inodal1 = Ynodal1.dot(Vnodal1)

    Ynodal2 = matrizesTransformadores2[ID]
    Vnodal2 = np.array([[V2[de][0]],[V2[para][0]]])
    Inodal2 = Ynodal2.dot(Vnodal2)
    
    Ynodal0 = matrizesTransformadores0[ID]
    Inodal0 = np.array([[0],[0]])
    
    if conexao in ['D-D','D-Y','Y-D','Y-Y','Yt-Y','Y-Yt']:
        continue
        
    elif conexao == 'Yt-D':
        Inodal0 = np.array([[Ynodal0*V0[de][0]],[-Ynodal0*V0[de][0]]])
    
    elif conexao == 'D-Yt':
        Inodal0 = np.array([[Ynodal0*V0[para][0]],[-Ynodal0*V0[para][0]]])
    
    elif conexao == 'Yt-Yt':
        Ynodal0 = matrizesTransformadores0[ID]
        Vnodal0 = np.array([[V0[de][0]],[V0[para][0]]])
        Inodal0 = Ynodal0.dot(Vnodal0)
        
    I012DePara = np.array([[Inodal0[0][0]],[Inodal1[0][0]],[Inodal2[0][0]]])
    I012ParaDe = np.array([[Inodal0[1][0]],[Inodal1[1][0]],[Inodal2[1][0]]])
    IabcDePara = T.dot(I012DePara)
    IabcParaDe = T.dot(I012ParaDe)

    aux.append([de,para,IabcDePara[0][0],IabcDePara[1][0],IabcDePara[2][0],IabcParaDe[0][0],IabcParaDe[1][0],IabcParaDe[2][0]])
    
contTransformadores = pd.DataFrame(data=aux,columns=['De','Para','Ia (De-Para)','Ib (De-Para)','Ic (De-Para)','Ia (Para-De)','Ib (Para-De)','Ic (Para-De)'])

aux = []
for ID in linhas.index:
    de = linhas.at[ID,'De']
    para = linhas.at[ID,'Para']
    
    Ynodal0 = matrizesLinhas0[ID]
    Vnodal0 = np.array([[V0[de][0]],[V0[para][0]]])
    Inodal0 = Ynodal0.dot(Vnodal0)
    
    Ynodal1 = matrizesLinhas1[ID]
    Vnodal1 = np.array([[V1[de][0]],[V1[para][0]]])
    Inodal1 = Ynodal1.dot(Vnodal1)

    Ynodal2 = matrizesLinhas2[ID]
    Vnodal2 = np.array([[V2[de][0]],[V2[para][0]]])
    Inodal2 = Ynodal2.dot(Vnodal2)
    
    I012DePara = np.array([[Inodal0[0][0]],[Inodal1[0][0]],[Inodal2[0][0]]])
    I012ParaDe = np.array([[Inodal0[1][0]],[Inodal1[1][0]],[Inodal2[1][0]]])
    IabcDePara = T.dot(I012DePara)
    IabcParaDe = T.dot(I012ParaDe)
    
    aux.append([de,para,IabcDePara[0][0],IabcDePara[1][0],IabcDePara[2][0],IabcParaDe[0][0],IabcParaDe[1][0],IabcParaDe[2][0]])
    
contLinhas = pd.DataFrame(data=aux,columns=['De','Para','Ia (De-Para)','Ib (De-Para)','Ic (De-Para)','Ia (Para-De)','Ib (Para-De)','Ic (Para-De)'])

aux = []
for ID in geradores.index:
    conexao = geradores.at[ID,'Conexao']
    barra = geradores.at[ID,'Barra']
    Y1 = geradores.at[ID,'Y1']
    Inodal1 = Y1*(1-V1[barra][0])
    Y2 = geradores.at[ID,'Y2']
    Inodal2 = Y2*(-V2[barra][0])
    Inodal0 = 0
    if conexao == 'Yt':
        Inodal0 = matrizesGeradores0[ID]*(-V0[barra][0])
    elif conexao in ['Y','D']:
        continue
    
    I012 = np.array([[Inodal0],[Inodal1],[Inodal2]])
    Iabc = T.dot(I012)
    
    aux.append([barra,Iabc[0][0],Iabc[1][0],Iabc[2][0]])
    
contGeradores = pd.DataFrame(data=aux,columns=['Barra','Ia','Ib','Ic'])

aux = []
for ID in tts.index:
    barra = tts.at[ID,'Barra']
    Y0 = tts.at[ID,'Y0']
    Inodal0 = -Y0*(V0[barra][0])
    
    I012 = np.array([[Inodal0],[0],[0]])
    Iabc = T.dot(I012)
    
    aux.append([barra,Iabc[0][0],Iabc[1][0],Iabc[2][0]])
    
contTts = pd.DataFrame(data=aux,columns=['Barra','Ia','Ib','Ic'])

print('Corrente de falta: ' + str(If) + '.')

print('Tensões nas barras:\n\n')
print(Tensoes)

print('\nContribuições dos transformadores\n\n')
print(contTransformadores)

print('\nContribuições das linhas\n\n')
print(contLinhas)

print('\nContribuições dos geradores\n\n')
print(contGeradores)

print('\nContribuições dos TTs\n\n')
print(contTts)
'''