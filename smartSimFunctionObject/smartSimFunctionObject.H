/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 AUTHOR, AFFILIATION
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::functionObjects::smartSimFunctionObject

Description
    Forward fields to Redis database through smartredis

\*---------------------------------------------------------------------------*/

#ifndef smartSimFunctionObject_H
#define smartSimFunctionObject_H

#include "fvMeshFunctionObject.H"
#include "client.h"
#include "wordList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                   Class smartSimFunctionObject Declaration
\*---------------------------------------------------------------------------*/

class smartSimFunctionObject
:
    public fvMeshFunctionObject
{
    // Private Data

        //- Set to false if not using a clustered database 
        bool clusterMode_;
        
        //- List of field names to send-receive from SmartRedis
        wordList fieldNames_;

        //- Single components of vector fields
        wordList fieldComponents_;

        //- List of field dimensions to send-receive from SmartRedis. In a
        //  list (1,3,6), 1 is scalar field dimension, 3 is a vector field
        //  dimension and 6 is a symmetric tensor field dimension.
        labelList fieldDimensions_;
        
        //- SmartRedis Database Client
        SmartRedis::Client client_;

public:

    //- Runtime type information
    TypeName("smartSimFunctionObject");


    // Constructors

        //- Construct from Time and dictionary
        smartSimFunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );

        //- No copy construct
        smartSimFunctionObject(const smartSimFunctionObject&) = delete;

        //- No copy assignment
        void operator=(const smartSimFunctionObject&) = delete;


    //- Destructor
    virtual ~smartSimFunctionObject() = default;


    // Member Functions
    
        virtual bool read(const dictionary& dict);

        virtual bool execute();

        virtual bool end();

        virtual bool write();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
